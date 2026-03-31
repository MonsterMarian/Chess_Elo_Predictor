import os
import asyncio
import aiohttp
import sqlite3
import time
import requests
from datetime import datetime
from chessdotcom import Client, get_leaderboards
from tqdm import tqdm

# --- Config ---
Client.request_config["headers"]["User-Agent"] = "EloPredictorCrawler/4.1 (contact: mvyst@example.com)"
HEADERS = Client.request_config["headers"]
DB_PATH = 'data/elo_history_v3.db'
CONCURRENT_REQUESTS = 6
RATE_LIMIT_SLEEP = 0.3

# --- Database ---
def init_db():
    if not os.path.exists('data'):
        os.makedirs('data')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Snapshots: one row per (player, month, time_class)
    # games_count: number of games played that month → activity feature #3
    c.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            username    TEXT,
            month_idx   INTEGER,
            elo         INTEGER,
            games_count INTEGER DEFAULT 1,
            time_class  TEXT,
            PRIMARY KEY (username, month_idx, time_class)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS players (
            username TEXT PRIMARY KEY,
            status   TEXT DEFAULT 'pending',
            join_ts  INTEGER
        )
    ''')
    conn.commit()
    return conn

def add_to_queue(conn, usernames):
    c = conn.cursor()
    for u in usernames:
        c.execute("INSERT OR IGNORE INTO players (username) VALUES (?)", (u.lower(),))
    conn.commit()

def get_next_pending(conn):
    c = conn.cursor()
    c.execute("SELECT username FROM players WHERE status='pending' LIMIT 1")
    row = c.fetchone()
    return row[0] if row else None

def set_status(conn, username, status):
    c = conn.cursor()
    c.execute("UPDATE players SET status=?, join_ts=join_ts WHERE username=?", (status, username.lower()))
    conn.commit()

def save_snapshots(conn, username, snapshots):
    """snapshots: list of (month_idx, elo, games_count, time_class)"""
    c = conn.cursor()
    for month_idx, elo, games_count, time_class in snapshots:
        c.execute(
            """INSERT OR REPLACE INTO snapshots (username, month_idx, elo, games_count, time_class)
               VALUES (?, ?, ?, ?, ?)""",
            (username.lower(), month_idx, elo, games_count, time_class)
        )
    conn.commit()

# --- Player info ---
def get_player_info(username):
    """Returns (current_elo, join_ts) or None if outside 600-2000."""
    try:
        r = requests.get(f"https://api.chess.com/pub/player/{username}/stats", headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return None
        stats = r.json()
        blitz  = stats.get('chess_blitz',  {}).get('last', {}).get('rating', 0)
        rapid  = stats.get('chess_rapid',  {}).get('last', {}).get('rating', 0)
        bullet = stats.get('chess_bullet', {}).get('last', {}).get('rating', 0)

        # Include player if ANY time control is in 600-2000 range
        ratings_in_range = [r for r in [blitz, rapid, bullet] if 600 <= r <= 2000]
        if not ratings_in_range:
            return None
        current_elo = max(ratings_in_range)  # representative elo for logging

        r2 = requests.get(f"https://api.chess.com/pub/player/{username}", headers=HEADERS, timeout=8)
        if r2.status_code != 200:
            return None
        join_ts = r2.json().get('joined', None)
        return current_elo, join_ts
    except Exception:
        return None

# --- Async archive fetching ---
async def fetch_archive(session, url, username, join_ts):
    """Fetches one monthly archive. Returns ({(month_idx, tc): [elos]}, opponents_set)."""
    result = {}
    opponents = set()
    try:
        async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            if resp.status != 200:
                return result, opponents
            data = await resp.json(content_type=None)
            for game in data.get('games', []):
                tc = game.get('time_class')
                if tc not in ('blitz', 'rapid', 'bullet'):
                    continue

                w_user = game['white']['username'].lower()
                b_user = game['black']['username'].lower()

                if w_user == username.lower():
                    elo = game['white'].get('rating')
                    opponents.add(game['black']['username'])
                elif b_user == username.lower():
                    elo = game['black'].get('rating')
                    opponents.add(game['white']['username'])
                else:
                    continue

                end_time = game.get('end_time')
                if elo and end_time and join_ts:
                    game_dt = datetime.fromtimestamp(end_time)
                    join_dt = datetime.fromtimestamp(join_ts)
                    month_idx = max(0, (game_dt.year - join_dt.year) * 12 + (game_dt.month - join_dt.month))
                    key = (month_idx, tc)
                    result.setdefault(key, []).append(elo)

    except asyncio.CancelledError:
        pass  # timeout cancelled — return whatever we have
    except Exception:
        pass
    return result, opponents

async def fetch_all_archives(archive_urls, username, join_ts):
    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch_archive(session, url, username, join_ts) for url in archive_urls]
            # return_exceptions=True prevents one timeout from aborting the whole batch
            results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out any exception objects
        return [r for r in results if isinstance(r, tuple)]
    except asyncio.CancelledError:
        return []

# --- Main player pipeline ---
def process_player(username, conn):
    info = get_player_info(username)
    if info is None:
        return False, set()

    current_elo, join_ts = info

    try:
        r = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives", headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return False, set()
        archive_urls = r.json().get('archives', [])
    except Exception:
        return False, set()

    if not archive_urls:
        return False, set()

    all_results = asyncio.run(fetch_all_archives(archive_urls, username, join_ts))

    # Aggregate: per (month_idx, tc) → last elo of the month, count of games
    aggregated = {}  # (month_idx, tc) → {'elos': [], 'count': 0}
    all_opponents = set()

    for month_data, opponents in all_results:
        all_opponents |= opponents
        for (month_idx, tc), elos in month_data.items():
            key = (month_idx, tc)
            if key not in aggregated:
                aggregated[key] = {'elos': [], 'count': 0}
            aggregated[key]['elos'].extend(elos)
            aggregated[key]['count'] += len(elos)

    if not aggregated:
        return False, set()

    snapshots = []
    for (month_idx, tc), data in aggregated.items():
        last_elo = data['elos'][-1]  # Last game of the month = most current rating
        games_count = data['count']
        snapshots.append((month_idx, last_elo, games_count, tc))

    save_snapshots(conn, username, snapshots)
    add_to_queue(conn, list(all_opponents))
    return True, all_opponents

# --- Seeding ---
def seed_queue(conn):
    print("Seeding queue from clubs and leaderboards...")
    seeds = set()
    clubs = ['chess-com', 'chess-com-university', 'casual-chess-club', 'team-usa', 'global-chess-league']
    for club in clubs:
        try:
            r = requests.get(f"https://api.chess.com/pub/club/{club}/members", headers=HEADERS, timeout=10)
            if r.status_code == 200:
                members = r.json()
                for cat in ['weekly', 'monthly', 'all_time']:
                    for m in members.get(cat, []):
                        seeds.add(m['username'])
                        if len(seeds) > 1000:
                            break
                    if len(seeds) > 1000:
                        break
        except Exception:
            continue
    try:
        lb = requests.get("https://api.chess.com/pub/leaderboards", headers=HEADERS, timeout=10).json()
        for t in ['live_blitz', 'live_rapid']:
            for p in lb.get(t, []):
                seeds.add(p['username'])
    except Exception:
        pass
    add_to_queue(conn, list(seeds))
    print(f"Queue seeded with {len(seeds)} players.")

# --- Crawler ---
def run_crawler(limit=5000):
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM players")
    if c.fetchone()[0] == 0:
        seed_queue(conn)

    pbar = tqdm(total=limit, desc="Crawling", unit="player")
    done_count = 0

    while done_count < limit:
        username = get_next_pending(conn)
        if not username:
            print("Queue empty.")
            break

        start = time.time()
        success, opponents = process_player(username, conn)
        elapsed = time.time() - start

        if success:
            set_status(conn, username, 'done')
            done_count += 1
            pbar.update(1)
            pbar.set_postfix({"player": username, "s": f"{elapsed:.1f}", "+queue": len(opponents)})
        else:
            set_status(conn, username, 'skipped')

        time.sleep(RATE_LIMIT_SLEEP)

    pbar.close()
    conn.close()

    conn2 = sqlite3.connect(DB_PATH)
    c2 = conn2.cursor()
    c2.execute("SELECT COUNT(DISTINCT username) FROM snapshots")
    players = c2.fetchone()[0]
    c2.execute("SELECT COUNT(*) FROM snapshots")
    snaps = c2.fetchone()[0]
    conn2.close()
    print(f"\n✅ Done. Players: {players} | Snapshots: {snaps} | DB: {DB_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5000)
    args = parser.parse_args()
    run_crawler(limit=args.limit)

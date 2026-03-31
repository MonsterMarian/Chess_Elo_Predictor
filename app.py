from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
from datetime import datetime
from predictor import EloPredictor
from chessdotcom import Client
import os

app = Flask(__name__)
predictor = EloPredictor()

HEADERS = dict(Client.request_config["headers"])
HEADERS["User-Agent"] = "EloPredictorCrawler/4.1 (contact: mvyst@example.com)"

# ------------------------------------------------------------------ #
# Chess.com API helpers                                                #
# ------------------------------------------------------------------ #

def get_player_stats_raw(username):
    """Fetch full stats JSON from Chess.com. Returns dict or None."""
    try:
        r = requests.get(f"https://api.chess.com/pub/player/{username}/stats",
                         headers=HEADERS, timeout=8)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def get_percentile(stats, time_class):
    """
    Extract Chess.com's built-in percentile metric (#9).
    Chess.com returns 'percentile' inside each rating type.
    """
    tc_map = {'blitz': 'chess_blitz', 'rapid': 'chess_rapid', 'bullet': 'chess_bullet'}
    key = tc_map.get(time_class, 'chess_blitz')
    return stats.get(key, {}).get('percentile', None)

# ------------------------------------------------------------------ #
# Async snapshot fetching (for live user lookup in app)               #
# ------------------------------------------------------------------ #

async def _fetch_archive(session, url, username, join_ts, time_class):
    snaps = []
    try:
        async with session.get(url, headers=HEADERS,
                               timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return snaps
            data = await resp.json(content_type=None)
            month_elos = []
            for game in data.get('games', []):
                tc = game.get('time_class')
                if tc != time_class:
                    continue
                if game['white']['username'].lower() == username.lower():
                    elo = game['white'].get('rating')
                elif game['black']['username'].lower() == username.lower():
                    elo = game['black'].get('rating')
                else:
                    continue
                end_time = game.get('end_time')
                if elo and end_time and join_ts:
                    game_dt = datetime.fromtimestamp(end_time)
                    join_dt = datetime.fromtimestamp(join_ts)
                    month_idx = max(0, (game_dt.year - join_dt.year) * 12 +
                                       (game_dt.month - join_dt.month))
                    month_elos.append({'month_idx': month_idx, 'elo': elo})
            if month_elos:
                by_month = {}
                for item in month_elos:
                    mi = item['month_idx']
                    by_month.setdefault(mi, []).append(item['elo'])
                for mi, elos in by_month.items():
                    snaps.append({
                        'month_idx': mi,
                        'elo': elos[-1],
                        'games_count': len(elos)
                    })
    except Exception:
        pass
    return snaps

async def get_player_snapshots(username, join_ts, time_class):
    try:
        r = requests.get(f"https://api.chess.com/pub/player/{username}/games/archives",
                         headers=HEADERS, timeout=8)
        if r.status_code != 200:
            return []
        urls = r.json().get('archives', [])
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8)) as session:
            tasks = [_fetch_archive(session, u, username, join_ts, time_class) for u in urls]
            results = await asyncio.gather(*tasks)
        return [s for month in results for s in month]
    except Exception:
        return []

# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data       = request.json
    username   = data.get('username', '').strip()
    time_class = data.get('type', 'blitz')
    target_days = int(data.get('target_days', 30))
    target_elo  = data.get('target_elo')

    if not username:
        return jsonify({'error': 'Username is required'}), 400
    if target_days < 1 or target_days > 90:
        return jsonify({'error': 'Počet dní pro predikci musí být mezi 1 a 90 dny (maximální spolehlivý výhled).'}), 400

    try:
        # 1. Get player profile + stats
        r_profile = requests.get(f"https://api.chess.com/pub/player/{username}",
                                  headers=HEADERS, timeout=8)
        if r_profile.status_code != 200:
            return jsonify({'error': f'Player "{username}" not found on Chess.com'}), 404

        profile  = r_profile.json()
        join_ts  = profile.get('joined')
        country  = profile.get('country', '').split('/')[-1]

        # 2. Stats + percentile (#9)
        stats_raw = get_player_stats_raw(username)
        percentile = None
        if stats_raw:
            percentile = get_percentile(stats_raw, time_class)

        # 3. Fetch historical snapshots (async)
        snaps = asyncio.run(get_player_snapshots(username, join_ts, time_class))
        if not snaps:
            return jsonify({'error': f'No {time_class} games found for "{username}".'}), 404

        df = pd.DataFrame(snaps).drop_duplicates('month_idx').sort_values('month_idx')

        if len(df) < 3:
            return jsonify({'error': f'Need at least 3 months of {time_class} data.'}), 400

        current_elo = int(df['elo'].iloc[-1])
        if current_elo < 600 or current_elo > 2000:
            return jsonify({'error': f'Omlouváme se, ale váš aktuální {time_class} rating ({current_elo}) je mimo náš spolehlivý rozsah. Aplikace dokáže přesně predikovat pouze pro hráče s Elo mezi 600 a 2000.'}), 400

        features    = predictor.build_features(df, time_class=time_class)

        # 4. Predict Elo after N days (convert to months)
        months_ahead = max(1, round(target_days / 30))
        pred_elo, pred_std = predictor.predict_elo_after_months(
            features, months_ahead=months_ahead, time_class=time_class)

        # Confidence label (#8)
        if pred_std < 5:
            confidence = "Vysoká"
        elif pred_std < 12:
            confidence = "Střední"
        else:
            confidence = "Nízká"

        # 5. Milestones
        m1, mo_m1, std_m1, msg_m1, m2, mo_m2, std_m2, msg_m2 = \
            predictor.predict_milestone(features, time_class=time_class)

        # 6. Custom target
        custom_result = None
        if target_elo:
            mo_c, std_c, msg_c = predictor.predict_time_to_target(
                features, int(target_elo), time_class)
            custom_result = {'target': int(target_elo), 'months': mo_c,
                             'std': std_c, 'msg': msg_c}

        return jsonify({
            'username': username,
            'country': country,
            'current_elo': current_elo,
            'predicted_elo': pred_elo,
            'pred_std': pred_std,
            'confidence': confidence,
            'target_days': target_days,
            'percentile': percentile,      # #9
            'milestone_1': {'target': m1, 'months': mo_m1, 'std': std_m1, 'msg': msg_m1},
            'milestone_2': {'target': m2, 'months': mo_m2, 'std': std_m2, 'msg': msg_m2},
            'custom': custom_result,
            'type': time_class
        })

    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Nepodařilo se připojit k Chess.com API (chyba sítě/DNS). Zkuste to prosím za chvíli.'}), 503
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Internal error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

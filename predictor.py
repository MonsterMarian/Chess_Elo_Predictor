import pandas as pd
import numpy as np
import sqlite3
import joblib
import os

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

DB_PATH = 'data/elo_history_v3.db'
MODEL_DIR = 'models'

# --- Elo bands for stratified models (#2) ---
ELO_BANDS = [
    (600,  1000, 'low'),
    (1000, 1400, 'mid_low'),
    (1400, 1800, 'mid_high'),
    (1800, 2100, 'high'),
]

def get_elo_band(elo):
    for lo, hi, label in ELO_BANDS:
        if lo <= elo < hi:
            return label
    return 'mid_low'  # fallback

def model_key(time_class, elo_band):
    return f"{time_class}_{elo_band}"


class EloPredictor:
    def __init__(self):
        self._models  = {}   # key → lgb model
        self._scalers = {}   # key → scaler (fallback k-NN)

    def _encode_tc(self, tc):
        return {'blitz': 0, 'rapid': 1, 'bullet': 2}.get(tc, 0)

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #
    def _build_training_pairs(self, group):
        """
        For a single (username, time_class) group (sorted by month_idx)
        returns list of (features, target_delta).
        Features: #2 stratification, #3 activity, #4 comeback/gap
        """
        group = group.drop_duplicates('month_idx').sort_values('month_idx')
        if len(group) < 4:
            return []

        elos   = group['elo'].values
        months = group['month_idx'].values
        counts = group['games_count'].values if 'games_count' in group.columns else np.ones(len(elos))
        peak   = np.max(elos)
        tc_enc = self._encode_tc(group['time_class'].iloc[0])

        pairs = []
        for i in range(3, len(elos) - 1):
            window_elo = elos[max(0, i-3):i+1]
            current_elo = elos[i]
            target_delta = elos[i+1] - current_elo

            # Feature #3: Activity (avg games/month in window)
            window_cnt = counts[max(0, i-3):i+1]
            avg_activity = np.mean(window_cnt)

            # Feature #4: Comeback / gap detection
            month_diffs = np.diff(months[max(0, i-3):i+1])
            max_gap = int(np.max(month_diffs)) if len(month_diffs) > 0 else 0

            growth_3m   = np.mean(np.diff(window_elo)) if len(window_elo) > 1 else 0
            volatility  = np.std(window_elo)
            elo_vs_peak = current_elo - peak

            features = [
                current_elo,
                int(months[i]),   # account age
                growth_3m,
                volatility,
                peak,
                elo_vs_peak,
                tc_enc,
                avg_activity,     # #3
                max_gap,          # #4
            ]
            pairs.append((get_elo_band(current_elo), features, target_delta))
        return pairs

    def prepare_features(self, df):
        """Returns dict: band → (X_array, y_array)"""
        df = df.sort_values(['username', 'time_class', 'month_idx'])
        band_data = {}

        for (username, tc), group in df.groupby(['username', 'time_class']):
            for band, feats, delta in self._build_training_pairs(group):
                if band not in band_data:
                    band_data[band] = ([], [])
                band_data[band][0].append(feats)
                band_data[band][1].append(delta)

        return {b: (np.array(X), np.array(y)) for b, (X, y) in band_data.items()}

    # ------------------------------------------------------------------ #
    # Training (#2: one model per band per time_class)                    #
    # ------------------------------------------------------------------ #
    def train(self, time_class=None):
        if not os.path.exists(DB_PATH):
            print(f"DB not found: {DB_PATH}")
            return False

        conn = sqlite3.connect(DB_PATH)
        q = "SELECT username, month_idx, elo, games_count, time_class FROM snapshots"
        params = []
        if time_class:
            q += " WHERE time_class = ?"
            params = [time_class]
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()

        if len(df) < 50:
            print(f"Not enough data ({len(df)} rows). Keep crawling.")
            return False

        band_features = self.prepare_features(df)
        if not band_features:
            print("No training pairs built.")
            return False

        os.makedirs(MODEL_DIR, exist_ok=True)
        trained = 0

        for band, (X, y) in band_features.items():
            if len(X) < 10:
                print(f"Band '{band}': too few samples ({len(X)}), skipping.")
                continue

            tc_label = time_class or 'all'
            key = f"{tc_label}_{band}"
            print(f"Training [{key}] on {len(X)} samples...")

            if LGBM_AVAILABLE:
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'min_data_in_leaf': max(5, len(X) // 20),
                    'verbose': -1,
                }
                train_data = lgb.Dataset(X, label=y)
                model = lgb.train(params, train_data, num_boost_round=300,
                                  callbacks=[lgb.log_evaluation(200)])
                self._models[key] = model
            else:
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_sc = scaler.fit_transform(X)
                model = KNeighborsRegressor(n_neighbors=min(5, len(X)))
                model.fit(X_sc, y)
                self._models[key] = model
                self._scalers[key] = scaler
                joblib.dump(scaler, f'{MODEL_DIR}/scaler_{key}.pkl')

            joblib.dump(model, f'{MODEL_DIR}/model_{key}.pkl')
            trained += 1

        print(f"✅ Trained {trained} models.")
        return trained > 0

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self, key):
        if key in self._models:
            return True
        path = f'{MODEL_DIR}/model_{key}.pkl'
        if os.path.exists(path):
            self._models[key] = joblib.load(path)
            sc_path = f'{MODEL_DIR}/scaler_{key}.pkl'
            if os.path.exists(sc_path):
                self._scalers[key] = joblib.load(sc_path)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Prediction                                                           #
    # ------------------------------------------------------------------ #
    def _predict_delta(self, features, key):
        """Predict monthly Elo delta. Also returns confidence (std across trees)."""
        if not self._ensure_loaded(key):
            raise RuntimeError(f"Model '{key}' not found. Run: python predictor.py --type {key.split('_')[0]}")

        X = np.array([features])
        model = self._models[key]

        if LGBM_AVAILABLE:
            # #8 Confidence: variance across individual trees
            leaf_pred = model.predict(X, num_iteration=model.best_iteration or 300)
            delta = float(leaf_pred[0])
            # Get variance as proxy for confidence using staged predictions
            staged = [model.predict(X, num_iteration=i)[0] for i in range(50, 301, 50)]
            confidence_std = float(np.std(staged))
        else:
            sc = self._scalers.get(key)
            X_sc = sc.transform(X) if sc else X
            delta = float(model.predict(X_sc)[0])
            confidence_std = 20.0  # Fixed uncertainty for k-NN fallback

        return delta, confidence_std

    def build_features(self, snap_df, time_class='blitz'):
        """
        snap_df: DataFrame with columns [month_idx, elo, (optional) games_count]
        Returns feature vector for prediction.
        """
        df = snap_df.sort_values('month_idx')
        elos   = df['elo'].values
        months = df['month_idx'].values
        counts = df['games_count'].values if 'games_count' in df.columns else np.ones(len(elos))

        peak   = float(np.max(elos))
        current_elo = float(elos[-1])
        tc_enc = self._encode_tc(time_class)

        window_elo = elos[-4:] if len(elos) >= 4 else elos
        window_cnt = counts[-4:] if len(counts) >= 4 else counts
        window_months = months[-4:] if len(months) >= 4 else months

        growth_3m    = float(np.mean(np.diff(window_elo))) if len(window_elo) > 1 else 0.0
        volatility   = float(np.std(window_elo))
        elo_vs_peak  = current_elo - peak
        avg_activity = float(np.mean(window_cnt))

        month_diffs = np.diff(window_months)
        max_gap = int(np.max(month_diffs)) if len(month_diffs) > 0 else 0

        return [current_elo, int(months[-1]), growth_3m, volatility,
                peak, elo_vs_peak, tc_enc, avg_activity, max_gap]

    def predict_elo_after_months(self, features, months_ahead=1, time_class='blitz'):
        """Simulate month-by-month growth. Returns (elo, cumulative_std)."""
        current = features[0]
        feat = list(features)
        cumulative_var = 0.0

        for _ in range(months_ahead):
            band = get_elo_band(current)
            key  = f"{time_class}_{band}"
            delta, std = self._predict_delta(feat, key)
            current += delta
            cumulative_var += std ** 2
            feat[0] = current
            feat[1] = feat[1] + 1   # advance month

        return round(current), round(np.sqrt(cumulative_var))

    def predict_time_to_target(self, features, target_elo, time_class='blitz'):
        """Simulate until target reached. Returns (months, cumulative_std, msg)."""
        current = features[0]
        if target_elo <= current:
            return 0, 0, "Již dosaženo"

        feat = list(features)
        cumulative_var = 0.0

        for month in range(1, 25):
            band = get_elo_band(current)
            key  = f"{time_class}_{band}"

            try:
                delta, std = self._predict_delta(feat, key)
            except RuntimeError:
                return None, None, "Model pro toto pásmo Elo nebyl nalezen."

            if delta <= 0:
                return None, None, "Trend nevede k cíli — rating stagnuje nebo klesá."

            current += delta
            cumulative_var += std ** 2
            feat[0] = current
            feat[1] = feat[1] + 1

            if current >= target_elo:
                return month, round(np.sqrt(cumulative_var)), "ok"

        return None, None, "Cíl překračuje spolehlivý horizont (2 roky)."

    def predict_milestone(self, features, time_class='blitz', offset=100):
        elo = features[0]
        m1  = (int(elo) // offset + 1) * offset
        m2  = m1 + offset
        months_m1, std_m1, msg_m1 = self.predict_time_to_target(features, m1, time_class)
        months_m2, std_m2, msg_m2 = self.predict_time_to_target(features, m2, time_class)
        return m1, months_m1, std_m1, msg_m1, m2, months_m2, std_m2, msg_m2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='blitz', choices=['blitz', 'rapid', 'bullet', 'all'])
    args = parser.parse_args()
    tc = None if args.type == 'all' else args.type
    print(f"Training for: {args.type} (all 4 Elo bands)...")
    EloPredictor().train(time_class=tc)

import sys
import pandas as pd
import numpy as np
from downloader import get_player_rating_history
from predictor import EloPredictor
import joblib

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('username', type=str)
    parser.add_argument('days', type=int)
    parser.add_argument('--type', type=str, default='blitz', choices=['blitz', 'rapid', 'bullet'])
    args = parser.parse_args()

    username = args.username
    days = args.days
    time_class = args.type
    
    print(f"--- Chess.com Elo Predictor ({time_class.capitalize()}) ---")
    print(f"Target: {username} after {days} days")
    
    # 1. Fetch user data (6 months is enough for current features)
    history = get_player_rating_history(username, months=6, time_class=time_class)
    if not history:
        print(f"Could not fetch {time_class} history for {username}.")
        return
        
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Upsample to daily
    daily = df.set_index('timestamp').resample('D').last().ffill()
    
    if len(daily) < 14:
        print(f"Not enough recent {time_class} history (need at least 14 days of activity).")
        return
        
    ratings = daily['elo'].values
    current_elo = ratings[-1]
    
    # 2. Extract features (Last 14 days)
    window = ratings[-14:]
    deltas = np.diff(window)
    mean_delta = np.mean(deltas) if len(deltas) > 0 else 0
    volatility = np.std(window)
    trend = (window[-1] - window[0]) / 14
    
    features = [current_elo, mean_delta, volatility, trend]
    
    # 3. Predict
    try:
        predictor = EloPredictor()
        # Train or load model for this specific time class
        predictor.train(time_class=time_class) 
        
        predicted_elo_7d = predictor.predict(features)
        delta_7d = predicted_elo_7d - current_elo
        
        # Scale delta based on requested days
        scaled_delta = delta_7d * (days / 7.0)
        final_prediction = current_elo + scaled_delta
        
        print("\n--- Results ---")
        print(f"Current Elo: {current_elo:.0f}")
        print(f"Predicted Elo in {days} days: {final_prediction:.0f}")
        print(f"Expected change: {scaled_delta:+.0f}")
        
        # Explain "Why?" using k-NN principle
        print("\nHow it works (k-NN):")
        print("The algorithm found players with similar recent rating trends,")
        print("volatility, and starting Elo. Their performance over the next")
        print("weeks was used to estimate your trajectory.")
        
    except Exception as e:
        print(f"Prediction error: {e}")

if __name__ == "__main__":
    main()

import joblib
from datetime import datetime
import numpy as np



# Chatbot Function
def predict_future_runs():
    print("\n--- Welcome to IPL Future Runs Predictor Chatbot ---\n")

    # Load model and encoders
    model = joblib.load('ipl_runs_predictor.pkl')
    label_encoders = joblib.load('label_encoders.pkl')

    # Collect user input
    player_name = input("Enter Player Name (e.g., Virat Kohli): ")
    stadium_name = input("Enter Stadium Name (e.g., Wankhede Stadium): ")
    match_date = input("Enter Match Date (YYYY-MM-DD): ")
    opposition_team = input("Enter Opposition Team (e.g., Mumbai Indians): ")

    # Process match_date
    try:
        match_date_parsed = datetime.strptime(match_date, '%Y-%m-%d')
        match_day = match_date_parsed.day
        match_month = match_date_parsed.month
        match_year = match_date_parsed.year
    except Exception as e:
        print("Invalid date format. Please enter date as YYYY-MM-DD.")
        return

    # Encode categorical inputs
    try:
        player_encoded = label_encoders['player_name'].transform([player_name])[0]
    except:
        player_encoded = 0  # Unknown player

    try:
        stadium_encoded = label_encoders['stadium_name'].transform([stadium_name])[0]
    except:
        stadium_encoded = 0  # Unknown stadium

    try:
        opposition_encoded = label_encoders['opposition_team'].transform([opposition_team])[0]
    except:
        opposition_encoded = 0  # Unknown opposition

    # Create input array
    input_data = np.array([[player_encoded, stadium_encoded, opposition_encoded, match_day, match_month, match_year]])

    # Predict
    predicted_runs = model.predict(input_data)[0]

    print(f"\n🏏 Predicted Runs: {predicted_runs:.2f}")

# 11. Run the chatbot
if __name__ == "__main__":
    predict_future_runs()

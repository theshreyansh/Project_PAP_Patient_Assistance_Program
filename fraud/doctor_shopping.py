import pandas as pd
from datetime import timedelta
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS

def detect_doctor_shopping(claims_df, patients_df):
    """
    Detect doctor shopping - patients visiting multiple doctors in a short period
    Returns: DataFrame with columns ['claim_id', 'type', 'description', 'score']
    """
    # Convert claim_date to datetime if it's not already
    claims_df = claims_df.copy()
    claims_df['claim_date'] = pd.to_datetime(claims_df['claim_date'])

    # Group by patient and count unique doctors within 30-day windows
    claims_df = claims_df.sort_values(['patient_id', 'claim_date'])

    # Create a function to count unique doctors in 30-day windows
    def count_doctors_in_window(group):
        group = group.sort_values('claim_date')
        doctor_counts = []
        for i, row in group.iterrows():
            window_start = row['claim_date']
            window_end = window_start + timedelta(days=30)
            window = group[(group['claim_date'] >= window_start) &
                         (group['claim_date'] <= window_end)]
            doctor_counts.append(len(window['doctor_id'].unique()))
        return pd.Series(doctor_counts, index=group.index)

    # Apply the function to each patient
    claims_df['doctor_count_30d'] = claims_df.groupby('patient_id').apply(count_doctors_in_window).reset_index(level=0, drop=True)

    # Find claims where patient saw >3 doctors in 30 days
    shopping = claims_df[claims_df['doctor_count_30d'] > FRAUD_THRESHOLDS.get('doctor_shopping', 3)]

    if shopping.empty:
        return pd.DataFrame(columns=['claim_id', 'type', 'description', 'score'])

    # Create result dataframe
    result = shopping[['claim_id']].drop_duplicates()
    result['type'] = 'doctor_shopping'
    result['description'] = f"Patient visited {shopping['doctor_count_30d'].max()} doctors in 30 days"
    result['score'] = np.minimum(shopping['doctor_count_30d'].max() * 0.2, 0.9)  # Scale score by doctor count

    return result[['claim_id', 'type', 'description', 'score']]
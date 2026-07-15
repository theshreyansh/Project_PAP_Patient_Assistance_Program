import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS

def detect_duplicate_claims(claims_df):
    """
    Detect duplicate claims - identical claims submitted multiple times
    Returns: DataFrame with columns ['claim_id', 'type', 'description', 'score']
    """
    # Create a key for grouping identical claims
    claims_df = claims_df.copy()
    claims_df['claim_key'] = (
        claims_df['patient_id'].astype(str) + "|" +
        claims_df['doctor_id'].astype(str) + "|" +
        claims_df['medication'].astype(str) + "|" +
        pd.to_datetime(claims_df['claim_date']).dt.strftime('%Y-%m-%d') + "|" +
        claims_df['total_amount'].astype(str)
    )

    # Find duplicate claim keys
    duplicate_keys = claims_df['claim_key'].value_counts()
    duplicate_keys = duplicate_keys[duplicate_keys > 1].index

    if len(duplicate_keys) == 0:
        return pd.DataFrame(columns=['claim_id', 'type', 'description', 'score'])

    # Get all claim IDs that are duplicates
    duplicate_claim_ids = claims_df[claims_df['claim_key'].isin(duplicate_keys)]['claim_id'].unique()

    # Create result dataframe
    result = pd.DataFrame({
        'claim_id': duplicate_claim_ids,
        'type': 'duplicate_claims',
        'description': 'Identical claim submitted multiple times',
        'score': 0.9  # High confidence in duplicate detection
    })

    return result
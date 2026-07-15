import pandas as pd
import numpy as np
from pathlib import Path
import sys
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS

def detect_vendor_collusion(claims_df, vendors_df, billing_df=None):
    """
    Detect vendor collusion - vendors working together to commit fraud
    """
    if billing_df is not None and {'vendor_id', 'claim_id'}.issubset(billing_df.columns):
        source_df = billing_df.merge(
            claims_df[['claim_id', 'patient_id', 'total_amount']],
            on='claim_id',
            how='left',
            suffixes=('', '_claim')
        )
        amount_col = 'total_amount'
    else:
        source_df = claims_df.copy()
        amount_col = 'total_amount'

    # Group by vendor and look for suspicious patterns
    if 'vendor_id' not in source_df.columns:
        return pd.DataFrame(columns=['vendor_id', 'score'])

    if 'patient_id' not in source_df.columns:
        source_df = source_df.copy()
        source_df['patient_id'] = np.nan

    aggregation_map = {
        'claim_id': 'count',
        amount_col: 'sum',
        'patient_id': 'nunique'
    }

    vendor_claims = source_df.groupby('vendor_id').agg(aggregation_map).reset_index()

    vendor_claims.columns = ['vendor_id', 'claim_count', 'total_amount', 'unique_patients']

    # Calculate average claim amount per vendor
    vendor_claims['avg_claim_amount'] = vendor_claims['total_amount'] / vendor_claims['claim_count']

    # Find vendors with:
    # 1. Very high claim volume
    # 2. Very high average claim amount
    # 3. Low number of unique patients (many claims for few patients)

    # Calculate thresholds
    claim_count_threshold = vendor_claims['claim_count'].quantile(0.95)
    amount_threshold = vendor_claims['avg_claim_amount'].quantile(0.95)
    patient_ratio_threshold = 0.1  # claims per unique patient

    # Find suspicious vendors
    suspicious_vendors = vendor_claims[
        (vendor_claims['claim_count'] > claim_count_threshold) |
        (vendor_claims['avg_claim_amount'] > amount_threshold) |
        (vendor_claims['claim_count'] / vendor_claims['unique_patients'] > 1/patient_ratio_threshold)
    ]

    if suspicious_vendors.empty:
        return pd.DataFrame(columns=['vendor_id', 'score'])

    # Keep the pairwise analysis bounded so the demo stays responsive.
    if len(suspicious_vendors) > 120:
        suspicious_vendors = suspicious_vendors.sort_values(
            ['claim_count', 'avg_claim_amount'],
            ascending=[False, False]
        ).head(120)

    # Find pairs of vendors that serve the same patients
    vendor_patient_matrix = source_df.pivot_table(
        index='vendor_id',
        columns='patient_id',
        values='claim_id',
        aggfunc='count',
        fill_value=0
    )

    # Find vendor pairs with overlapping patients
    colluding_pairs = []
    vendor_ids = suspicious_vendors['vendor_id'].tolist()

    for v1, v2 in combinations(vendor_ids, 2):
        if v1 in vendor_patient_matrix.index and v2 in vendor_patient_matrix.index:
            # Calculate Jaccard similarity of patient sets
            patients_v1 = set(vendor_patient_matrix.loc[v1][vendor_patient_matrix.loc[v1] > 0].index)
            patients_v2 = set(vendor_patient_matrix.loc[v2][vendor_patient_matrix.loc[v2] > 0].index)

            if len(patients_v1) > 0 and len(patients_v2) > 0:
                intersection = len(patients_v1 & patients_v2)
                union = len(patients_v1 | patients_v2)
                similarity = intersection / union if union > 0 else 0

                if similarity > FRAUD_THRESHOLDS.get('vendor_collusion', 0.5):
                    colluding_pairs.append({
                        'vendor_id_1': v1,
                        'vendor_id_2': v2,
                        'similarity': similarity,
                        'shared_patients': intersection
                    })

    # Create result dataframe
    result_vendors = set(suspicious_vendors['vendor_id'])

    if colluding_pairs:
        for pair in colluding_pairs:
            result_vendors.add(pair['vendor_id_1'])
            result_vendors.add(pair['vendor_id_2'])

    result = pd.DataFrame({
        'vendor_id': list(result_vendors),
        'type': 'vendor_collusion',
        'description': 'Potential vendor collusion - suspicious billing patterns',
        'score': 0.7  # Base score
    })

    # Increase score for vendors with multiple indicators
    high_volume = set(suspicious_vendors[suspicious_vendors['claim_count'] > claim_count_threshold]['vendor_id'])
    high_amount = set(suspicious_vendors[suspicious_vendors['avg_claim_amount'] > amount_threshold]['vendor_id'])

    result.loc[result['vendor_id'].isin(high_volume), 'score'] += 0.1
    result.loc[result['vendor_id'].isin(high_amount), 'score'] += 0.1

    # Cap score at 1.0
    result['score'] = np.minimum(result['score'], 1.0)

    return result

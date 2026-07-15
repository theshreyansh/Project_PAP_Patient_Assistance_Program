import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS, PAP_ELIGIBILITY_CRITERIA

def detect_eligibility_fraud(eligibility_df, patients_df, claims_df=None):
    """
    Detect eligibility fraud - patients misrepresenting their eligibility
    """
    # Merge with patients to get full information
    merged = eligibility_df.merge(patients_df, on='patient_id', how='left')

    # 1. Find patients who were denied but have approved claims
    denied_but_approved = merged[
        (merged['status'] == 'Denied') &
        (merged['is_eligible_for_pap'] == True)
    ]

    # 2. Find patients with income just below the threshold
    income_threshold = PAP_ELIGIBILITY_CRITERIA['income_threshold']
    income_mapping = {
        '< $25k': 15000, '$25k-$50k': 37500, '$50k-$75k': 62500,
        '$75k-$100k': 87500, '> $100k': 120000
    }
    merged['income_value'] = merged['income_level'].map(income_mapping)

    borderline_income = merged[
        (merged['income_value'] > income_threshold * 0.9) &
        (merged['income_value'] <= income_threshold) &
        (merged['is_eligible_for_pap'] == True)
    ]

    # 3. Find patients with insurance but claiming to be uninsured
    insurance_fraud = merged[
        (merged['insurance_type'] != 'Uninsured') &
        (merged['insurance_type'] != 'Underinsured') &
        (merged['is_eligible_for_pap'] == True)
    ]

    # 4. Find patients with multiple eligibility applications with different information
    app_counts = eligibility_df['patient_id'].value_counts()
    multiple_apps = app_counts[app_counts > 1].index
    inconsistent_apps = eligibility_df[
        eligibility_df['patient_id'].isin(multiple_apps)
    ].groupby('patient_id').filter(
        lambda x: x['income_verified'].nunique() > 1 or
                 x['insurance_verified'].nunique() > 1
    )

    # Combine all suspicious cases
    suspicious_patients = set()

    if not denied_but_approved.empty:
        suspicious_patients.update(denied_but_approved['patient_id'].unique())

    if not borderline_income.empty:
        suspicious_patients.update(borderline_income['patient_id'].unique())

    if not insurance_fraud.empty:
        suspicious_patients.update(insurance_fraud['patient_id'].unique())

    if not inconsistent_apps.empty:
        suspicious_patients.update(inconsistent_apps['patient_id'].unique())

    if not suspicious_patients:
        return pd.DataFrame(columns=['claim_id', 'score'])

    # Get claim IDs for these patients
    if claims_df is None:
        claims_df = pd.read_csv(Path(__file__).parent.parent / "data" / "claims.csv")
    suspicious_claims = claims_df[
        claims_df['patient_id'].isin(suspicious_patients)
    ]['claim_id'].unique()

    # Create result dataframe
    result = pd.DataFrame({
        'claim_id': suspicious_claims,
        'type': 'eligibility_fraud',
        'description': 'Potential eligibility fraud - misrepresented qualification criteria',
        'score': 0.8  # Base score
    })

    return result

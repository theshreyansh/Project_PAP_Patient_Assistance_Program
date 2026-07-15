import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS, PAP_MEDICATIONS

def detect_medication_diversion(prescriptions_df, claims_df, medications_df=None):
    """
    Detect medication diversion - unusual patterns in medication prescribing
    """
    # Link prescriptions to claims by patient/doctor/date proximity because the
    # demo data keeps prescriptions and claims in separate tables.
    merged = prescriptions_df.copy()
    if medications_df is not None and {'medication_id', 'name'}.issubset(medications_df.columns) and 'medication_id' in merged.columns:
        merged = merged.merge(
            medications_df[['medication_id', 'name', 'is_pap_medication']],
            on='medication_id',
            how='left',
            suffixes=('', '_med')
        )
        merged['medication_name'] = merged['name'].fillna(merged['medication_id'].astype(str))
    else:
        merged['medication_name'] = merged['medication'] if 'medication' in merged.columns else merged['medication_id'].astype(str)

    merged['prescription_date'] = pd.to_datetime(merged['prescription_date'], errors='coerce')
    claims_link = claims_df[['claim_id', 'patient_id', 'doctor_id', 'claim_date', 'is_pap_claim']].copy()
    claims_link['claim_date'] = pd.to_datetime(claims_link['claim_date'], errors='coerce')
    merged = merged.merge(
        claims_link,
        on=['patient_id', 'doctor_id'],
        how='left',
        suffixes=('', '_claim')
    )
    merged['date_gap'] = (merged['claim_date'] - merged['prescription_date']).abs().dt.days
    merged = merged[(merged['date_gap'].isna()) | (merged['date_gap'] <= 30)]

    # Focus on PAP medications
    pap_medications = set(PAP_MEDICATIONS)
    pap_prescriptions = merged[merged['medication_name'].isin(pap_medications)]

    if pap_prescriptions.empty:
        return pd.DataFrame(columns=['claim_id', 'score'])

    # 1. Find patients with multiple prescriptions for the same PAP medication
    med_patient_counts = pap_prescriptions.groupby(['patient_id', 'medication_name']).size().reset_index(name='count')
    suspicious_meds = med_patient_counts[med_patient_counts['count'] > 1]

    # 2. Find doctors prescribing PAP medications to many patients
    doctor_pap_counts = pap_prescriptions.groupby('doctor_id').size().reset_index(name='pap_prescription_count')
    suspicious_doctors = doctor_pap_counts[
        doctor_pap_counts['pap_prescription_count'] > doctor_pap_counts['pap_prescription_count'].quantile(0.95)
    ]

    # 3. Find prescriptions with unusually high quantities
    medication_stats = pap_prescriptions.groupby('medication_name')['quantity'].agg(['mean', 'std']).reset_index()
    merged_stats = pap_prescriptions.merge(medication_stats, on='medication_name')
    high_quantity = merged_stats[
        (merged_stats['quantity'] - merged_stats['mean']) / merged_stats['std'] > FRAUD_THRESHOLDS.get('medication_diversion', 1.5)
    ]

    # Combine all suspicious claims
    suspicious_claims = set()

    # From multiple prescriptions
    if not suspicious_meds.empty:
        suspicious_claims.update(
            pap_prescriptions[
                pap_prescriptions.apply(
                    lambda x: (x['patient_id'], x['medication_name']) in
                    list(zip(suspicious_meds['patient_id'], suspicious_meds['medication_name'])),
                    axis=1
                )
            ]['claim_id'].dropna().unique()
        )

    # From suspicious doctors
    if not suspicious_doctors.empty:
        suspicious_claims.update(
            pap_prescriptions[
                pap_prescriptions['doctor_id'].isin(suspicious_doctors['doctor_id'])
            ]['claim_id'].dropna().unique()
        )

    # From high quantities
    if not high_quantity.empty:
        suspicious_claims.update(high_quantity['claim_id'].dropna().unique())

    if not suspicious_claims:
        return pd.DataFrame(columns=['claim_id', 'score'])

    # Create result dataframe
    result = pd.DataFrame({
        'claim_id': list(suspicious_claims),
        'type': 'medication_diversion',
        'description': 'Potential medication diversion - unusual prescribing patterns',
        'score': 0.75  # Base score
    })

    return result

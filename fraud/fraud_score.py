import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS, DATA_DIR


def calculate_fraud_scores(
    claims_df,
    patients_df,
    doctors_df,
    vendors_df,
    medications_df,
    prescriptions_df,
    eligibility_df,
    pap_df,
    billing_df=None,
    progress_callback=None,
):
    """
    Calculate composite fraud scores for all entities.
    """
    from fraud.duplicate_claims import detect_duplicate_claims
    from fraud.doctor_shopping import detect_doctor_shopping
    from fraud.ghost_patients import detect_ghost_patients
    from fraud.vendor_collusion import detect_vendor_collusion
    from fraud.medication_diversion import detect_medication_diversion
    from fraud.eligibility_fraud import detect_eligibility_fraud

    claims_df = claims_df.copy()
    patients_df = patients_df.copy()
    doctors_df = doctors_df.copy()
    vendors_df = vendors_df.copy()

    for frame in (claims_df, patients_df, doctors_df, vendors_df):
        if 'fraud_score' not in frame.columns:
            frame['fraud_score'] = 0.0
        if 'fraud_flag' not in frame.columns:
            frame['fraud_flag'] = False

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)
        else:
            print(message)

    report("Detecting duplicate claims...")
    duplicate_claims = detect_duplicate_claims(claims_df)

    report("Detecting doctor shopping...")
    doctor_shopping = detect_doctor_shopping(claims_df, patients_df)

    report("Detecting ghost patient patterns...")
    ghost_patients = detect_ghost_patients(patients_df, claims_df)

    report("Detecting vendor collusion...")
    vendor_collusion = detect_vendor_collusion(claims_df, vendors_df, billing_df)

    report("Detecting medication diversion...")
    medication_diversion = detect_medication_diversion(prescriptions_df, claims_df, medications_df)

    report("Detecting eligibility misrepresentation...")
    eligibility_fraud = detect_eligibility_fraud(eligibility_df, patients_df, claims_df)

    if not duplicate_claims.empty:
        mask = claims_df['claim_id'].isin(duplicate_claims['claim_id'])
        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += duplicate_claims['score'].max()

    if not doctor_shopping.empty:
        mask = claims_df['claim_id'].isin(doctor_shopping['claim_id'])
        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += doctor_shopping['score'].max()

    if not ghost_patients.empty:
        mask = claims_df['patient_id'].isin(ghost_patients['patient_id'])
        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += ghost_patients['score'].max()

    if not vendor_collusion.empty:
        if billing_df is not None and 'vendor_id' in billing_df.columns:
            vendor_claim_ids = billing_df[billing_df['vendor_id'].isin(vendor_collusion['vendor_id'])]['claim_id'].unique()
            mask = claims_df['claim_id'].isin(vendor_claim_ids)
        elif 'vendor_id' in claims_df.columns:
            mask = claims_df['vendor_id'].isin(vendor_collusion['vendor_id'])
        else:
            mask = pd.Series(False, index=claims_df.index)

        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += vendor_collusion['score'].max()

    if not medication_diversion.empty:
        mask = claims_df['claim_id'].isin(medication_diversion['claim_id'])
        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += medication_diversion['score'].max()

    if not eligibility_fraud.empty:
        mask = claims_df['claim_id'].isin(eligibility_fraud['claim_id'])
        claims_df.loc[mask, 'fraud_flag'] = True
        claims_df.loc[mask, 'fraud_score'] += eligibility_fraud['score'].max()

    max_score = claims_df['fraud_score'].max()
    if max_score > 0:
        claims_df['fraud_score'] = claims_df['fraud_score'] / max_score
    report("Normalizing entity risk scores...")

    patient_fraud = claims_df.groupby('patient_id')['fraud_score'].agg(['mean', 'count']).reset_index()
    patient_fraud.columns = ['patient_id', 'avg_fraud_score', 'claim_count']
    patient_fraud['fraud_score'] = patient_fraud['avg_fraud_score'] * np.minimum(patient_fraud['claim_count'] / 10, 1)
    patients_df = patients_df.drop(columns=['fraud_score'], errors='ignore').merge(
        patient_fraud[['patient_id', 'fraud_score']],
        on='patient_id',
        how='left'
    )
    patients_df['fraud_score'] = patients_df['fraud_score'].fillna(0)
    patients_df['fraud_flag'] = patients_df['fraud_score'] > FRAUD_THRESHOLDS.get('patient_fraud', 0.5)

    doctor_fraud = claims_df.groupby('doctor_id')['fraud_score'].agg(['mean', 'count']).reset_index()
    doctor_fraud.columns = ['doctor_id', 'avg_fraud_score', 'claim_count']
    doctor_fraud['fraud_score'] = doctor_fraud['avg_fraud_score'] * np.minimum(doctor_fraud['claim_count'] / 20, 1)
    doctors_df = doctors_df.drop(columns=['fraud_score'], errors='ignore').merge(
        doctor_fraud[['doctor_id', 'fraud_score']],
        on='doctor_id',
        how='left'
    )
    doctors_df['fraud_score'] = doctors_df['fraud_score'].fillna(0)
    doctors_df['fraud_flag'] = doctors_df['fraud_score'] > FRAUD_THRESHOLDS.get('doctor_fraud', 0.5)

    if billing_df is not None and 'vendor_id' in billing_df.columns:
        vendor_source = billing_df.merge(claims_df[['claim_id', 'fraud_score']], on='claim_id', how='left')
        vendor_fraud = vendor_source.groupby('vendor_id')['fraud_score'].agg(['mean', 'count']).reset_index()
    elif 'vendor_id' in claims_df.columns:
        vendor_fraud = claims_df.groupby('vendor_id')['fraud_score'].agg(['mean', 'count']).reset_index()
    else:
        vendor_fraud = pd.DataFrame(columns=['vendor_id', 'avg_fraud_score', 'claim_count'])

    if not vendor_fraud.empty:
        vendor_fraud.columns = ['vendor_id', 'avg_fraud_score', 'claim_count']
        vendor_fraud['fraud_score'] = vendor_fraud['avg_fraud_score'] * np.minimum(vendor_fraud['claim_count'] / 15, 1)
        vendors_df = vendors_df.drop(columns=['fraud_score'], errors='ignore').merge(
            vendor_fraud[['vendor_id', 'fraud_score']],
            on='vendor_id',
            how='left'
        )

    vendors_df['fraud_score'] = vendors_df['fraud_score'].fillna(0)
    vendors_df['fraud_flag'] = vendors_df['fraud_score'] > FRAUD_THRESHOLDS.get('vendor_fraud', 0.5)
    report("Consolidating findings for executive review...")

    return {
        'claims': claims_df,
        'patients': patients_df,
        'doctors': doctors_df,
        'vendors': vendors_df,
        'duplicate_claims': duplicate_claims,
        'doctor_shopping': doctor_shopping,
        'ghost_patients': ghost_patients,
        'vendor_collusion': vendor_collusion,
        'medication_diversion': medication_diversion,
        'eligibility_fraud': eligibility_fraud,
    }

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent.parent))
from config import FRAUD_THRESHOLDS


def _normalize_text(value):
    tokens = re.findall(r"[A-Za-z0-9]+", str(value).lower())
    return " ".join(tokens)

def detect_ghost_patients(patients_df, claims_df):
    """
    Detect ghost patients - patients with suspicious identities or patterns
    """
    similar_patients = []
    normalized = patients_df.copy()
    normalized['normalized_name'] = (
        normalized.get('full_name', pd.Series(dtype=str)).astype(str).map(_normalize_text)
    )
    normalized['dob_value'] = pd.to_datetime(normalized.get('date_of_birth'), errors='coerce')

    # 1. Exact SSN collisions
    if 'ssn' in normalized.columns:
        for _, group in normalized.groupby('ssn'):
            if len(group) > 1:
                patient_ids = group['patient_id'].tolist()
                for i in range(len(patient_ids)):
                    for j in range(i + 1, len(patient_ids)):
                        similar_patients.append({
                            'patient_id_1': patient_ids[i],
                            'patient_id_2': patient_ids[j],
                            'similarity': 1.0,
                        })

    # 2. Exact name and DOB collisions
    for _, group in normalized.groupby(['normalized_name', 'dob_value']):
        if len(group) > 1:
            patient_ids = group['patient_id'].tolist()
            for i in range(len(patient_ids)):
                for j in range(i + 1, len(patient_ids)):
                    similar_patients.append({
                        'patient_id_1': patient_ids[i],
                        'patient_id_2': patient_ids[j],
                        'similarity': 0.95,
                    })

    # 3. Find patients with very high claim volume
    claim_counts = claims_df['patient_id'].value_counts()
    high_volume_patients = claim_counts[claim_counts > claim_counts.quantile(0.99)].index

    # 4. Find patients with no diagnosis
    no_diagnosis = patients_df[patients_df['diagnosis_code'].isna()]['patient_id']

    # Combine all suspicious patients
    suspicious_patients = set()
    for item in similar_patients:
        suspicious_patients.add(item['patient_id_1'])
        suspicious_patients.add(item['patient_id_2'])
    suspicious_patients.update(high_volume_patients)
    suspicious_patients.update(no_diagnosis)

    if not suspicious_patients:
        return pd.DataFrame(columns=['patient_id', 'score'])

    # Create result dataframe
    result = pd.DataFrame({
        'patient_id': list(suspicious_patients),
        'type': 'ghost_patients',
        'description': 'Potential ghost patient - suspicious identity or patterns',
        'score': 0.8  # Base score for ghost patients
    })

    # Increase score for patients with multiple indicators
    indicator_counts = {}
    for item in similar_patients:
        indicator_counts[item['patient_id_1']] = indicator_counts.get(item['patient_id_1'], 0) + 1
        indicator_counts[item['patient_id_2']] = indicator_counts.get(item['patient_id_2'], 0) + 1

    high_volume_set = set(high_volume_patients)
    no_diagnosis_set = set(no_diagnosis)

    result.loc[result['patient_id'].isin(high_volume_set), 'score'] += 0.1
    result.loc[result['patient_id'].isin(no_diagnosis_set), 'score'] += 0.1
    result['score'] = result['score'] + result['patient_id'].map(lambda pid: min(indicator_counts.get(pid, 0) * 0.05, 0.2))

    # Cap score at 1.0
    result['score'] = np.minimum(result['score'], 1.0)

    return result

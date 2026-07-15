from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, GENERATED_DATA_SIZE, PAP_ELIGIBILITY_CRITERIA

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_eligibility(patients_df, n=GENERATED_DATA_SIZE//2):
    """Generate realistic eligibility data for Kaiser PAP"""
    eligibility_statuses = ['Approved', 'Denied', 'Pending', 'Expired']
    denial_reasons = [
        'Income exceeds threshold', 'Incomplete application',
        'Not a US resident', 'Has insurance coverage',
        'Missing documentation', 'Diagnosis not covered'
    ]

    patient_ids = patients_df['patient_id'].tolist()

    data = []
    for i in range(n):
        patient_id = random.choice(patient_ids)
        patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]

        application_date = fake.date_between(start_date='-2y', end_date='today')

        income = patient['income_level']
        insurance = patient['insurance_type']

        income_value = {
            '< $25k': 15000, '$25k-$50k': 37500, '$50k-$75k': 62500,
            '$75k-$100k': 87500, '> $100k': 120000
        }.get(income, 50000)

        meets_income = income_value <= PAP_ELIGIBILITY_CRITERIA['income_threshold']
        meets_insurance = insurance in PAP_ELIGIBILITY_CRITERIA['insurance_status']
        meets_diagnosis = patient['diagnosis_code'] in PAP_ELIGIBILITY_CRITERIA['diagnosis_codes']
        is_eligible = meets_income and meets_insurance and meets_diagnosis

        if is_eligible:
            status = random.choices(['Approved', 'Pending', 'Expired'], weights=[0.7, 0.2, 0.1])[0]
            denial_reason = None
        else:
            status = random.choices(['Denied', 'Pending'], weights=[0.8, 0.2])[0]
            denial_reason = random.choice(denial_reasons) if status == 'Denied' else None

        expiration_date = application_date + timedelta(days=365) if status == 'Approved' else None

        data.append({
            'eligibility_id': f"ELG{1000000 + i}",
            'patient_id': patient_id,
            'application_date': application_date,
            'status': status,
            'denial_reason': denial_reason,
            'income_verified': income_value,
            'insurance_verified': insurance,
            'diagnosis_verified': patient['diagnosis_code'],
            'is_eligible': is_eligible,
            'expiration_date': expiration_date,
            'reviewed_by': f"REV{random.randint(1000, 9999)}",
            'review_date': application_date + timedelta(days=random.randint(1, 14)) if status != 'Pending' else None,
            'fraud_flag': False,
            'notes': fake.sentence() if random.random() < 0.2 else None,
            'created_at': fake.date_between(start_date='-2y', end_date='today')
        })

    df = pd.DataFrame(data)
    approved_mask = df['status'] == 'Approved'
    expired_mask = np.random.random(len(df)) < 0.1
    df.loc[approved_mask & expired_mask, 'status'] = 'Expired'
    df.loc[approved_mask & expired_mask, 'expiration_date'] = df.loc[approved_mask & expired_mask, 'expiration_date'] - timedelta(days=random.randint(1, 180))

    return df

if __name__ == "__main__":
    from generator.patient_generator import generate_patients
    patients = generate_patients()
    eligibility = generate_eligibility(patients)
    eligibility.to_csv(DATA_DIR / "eligibility.csv", index=False)
    print(f"Generated {len(eligibility)} eligibility records")
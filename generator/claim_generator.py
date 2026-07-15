from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, GENERATED_DATA_SIZE, PAP_MEDICATIONS

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_claims(patients_df, doctors_df, n=GENERATED_DATA_SIZE):
    claim_types = ['Pharmacy', 'Medical', 'Hospital', 'DME', 'Lab']
    claim_statuses = ['Submitted', 'Approved', 'Rejected', 'Pending', 'Paid']
    rejection_reasons = [
        'Incomplete documentation', 'Not medically necessary',
        'Patient not eligible', 'Duplicate claim', 'Exceeded limit',
        'Missing prior authorization', 'Expired coverage'
    ]

    patient_ids = patients_df['patient_id'].tolist()
    doctor_ids = doctors_df['doctor_id'].tolist()

    data = []
    for i in range(n):
        patient_id = random.choice(patient_ids)
        patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]

        same_state_doctors = doctors_df[doctors_df['state'] == patient['state']]
        if len(same_state_doctors) > 0 and random.random() < 0.8:
            doctor = same_state_doctors.sample(1).iloc[0]
        else:
            doctor = doctors_df.sample(1).iloc[0]

        claim_date = fake.date_between(start_date='-2y', end_date='today')
        processed_date = claim_date + timedelta(days=random.randint(1, 30))

        is_pap_claim = patient['is_eligible_for_pap'] and random.random() < 0.7

        if is_pap_claim:
            medication = random.choice(PAP_MEDICATIONS)
            amount = random.uniform(1000, 50000)
            quantity = random.randint(1, 12)
        else:
            medication = fake.word().capitalize() + str(random.randint(100, 999))
            amount = random.uniform(50, 5000)
            quantity = random.randint(1, 30)

        if is_pap_claim:
            status = random.choices(
                ['Approved', 'Paid', 'Submitted', 'Pending'],
                weights=[0.4, 0.3, 0.2, 0.1]
            )[0]
        else:
            status = random.choices(
                ['Approved', 'Paid', 'Submitted', 'Pending', 'Rejected'],
                weights=[0.3, 0.3, 0.2, 0.1, 0.1]
            )[0]

        data.append({
            'claim_id': f"CLM{10000000 + i}",
            'patient_id': patient_id,
            'doctor_id': doctor['doctor_id'],
            'claim_type': random.choice(claim_types),
            'claim_date': claim_date,
            'processed_date': processed_date,
            'medication': medication,
            'diagnosis_code': patient['diagnosis_code'],
            'quantity': quantity,
            'unit_price': round(amount / quantity, 2),
            'total_amount': round(amount, 2),
            'is_pap_claim': is_pap_claim,
            'pap_program': 'Kaiser PAP' if is_pap_claim else None,
            'status': status,
            'rejection_reason': random.choice(rejection_reasons) if status == 'Rejected' else None,
            'fraud_flag': False,
            'fraud_score': 0.0,
            'notes': fake.sentence() if random.random() < 0.2 else None,
            'created_at': fake.date_between(start_date='-2y', end_date='today')
        })

    df = pd.DataFrame(data)
    duplicate_mask = np.random.random(len(df)) < 0.1
    df.loc[duplicate_mask, 'claim_id'] = df.loc[duplicate_mask, 'claim_id'] + "_DUP"

    high_amount_mask = np.random.random(len(df)) < 0.05
    df.loc[high_amount_mask, 'total_amount'] = df.loc[high_amount_mask, 'total_amount'] * random.uniform(2, 5)

    return df

if __name__ == "__main__":
    from generator.patient_generator import generate_patients
    from generator.doctor_generator import generate_doctors

    patients = generate_patients()
    doctors = generate_doctors()
    claims = generate_claims(patients, doctors)
    claims.to_csv(DATA_DIR / "claims.csv", index=False)
    print(f"Generated {len(claims)} claim records")
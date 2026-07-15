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

def generate_pap_data(patients_df, medications_df, n=GENERATED_DATA_SIZE//10):
    """Generate Patient Assistance Program specific data for Kaiser"""
    pap_statuses = ['Active', 'Pending', 'Denied', 'Completed', 'Terminated']
    denial_reasons = [
        'Income verification failed', 'Insurance coverage found',
        'Documentation incomplete', 'Diagnosis not covered',
        'Patient deceased', 'Moved out of service area'
    ]

    eligible_patients = patients_df[patients_df['is_eligible_for_pap']]['patient_id'].tolist()
    pap_medications = medications_df[medications_df['is_pap_medication']]['medication_id'].tolist()

    data = []
    for i in range(n):
        patient_id = random.choice(eligible_patients)
        medication_id = random.choice(pap_medications)

        patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]
        medication = medications_df[medications_df['medication_id'] == medication_id].iloc[0]

        enrollment_date = fake.date_between(start_date='-2y', end_date='today')
        status = random.choices(
            ['Active', 'Pending', 'Completed', 'Denied', 'Terminated'],
            weights=[0.5, 0.2, 0.15, 0.1, 0.05]
        )[0]

        if status == 'Active':
            start_date = enrollment_date + timedelta(days=random.randint(1, 30))
            end_date = None
        elif status == 'Completed':
            start_date = enrollment_date + timedelta(days=random.randint(1, 30))
            end_date = start_date + timedelta(days=random.randint(30, 365))
        elif status == 'Denied':
            start_date = None
            end_date = None
        else:  # Terminated
            start_date = enrollment_date + timedelta(days=random.randint(1, 30))
            end_date = start_date + timedelta(days=random.randint(30, 180))

        data.append({
            'pap_id': f"PAP{1000000 + i}",
            'patient_id': patient_id,
            'medication_id': medication_id,
            'enrollment_date': enrollment_date,
            'start_date': start_date,
            'end_date': end_date,
            'status': status,
            'denial_reason': random.choice(denial_reasons) if status == 'Denied' else None,
            'monthly_supply': random.randint(1, medication['max_monthly_quantity']),
            'copay_amount': round(random.uniform(0, 50), 2),
            'shipment_frequency': random.choice(['Monthly', 'Quarterly', 'Biweekly']),
            'shipping_address': patient['address'],
            'shipping_city': patient['city'],
            'shipping_state': patient['state'],
            'shipping_zip': patient['zip_code'],
            'preferred_pharmacy': fake.company() + " Pharmacy",
            'case_manager': f"CM{random.randint(1000, 9999)}",
            'notes': fake.sentence() if random.random() < 0.2 else None,
            'fraud_flag': False,
            'created_at': fake.date_between(start_date='-2y', end_date='today')
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    from generator.patient_generator import generate_patients
    from generator.medication_generator import generate_medications

    patients = generate_patients()
    medications = generate_medications()
    pap = generate_pap_data(patients, medications)
    pap.to_csv(DATA_DIR / "pap.csv", index=False)
    print(f"Generated {len(pap)} PAP records")
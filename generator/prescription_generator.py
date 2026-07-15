from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, GENERATED_DATA_SIZE

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_prescriptions(patients_df, doctors_df, medications_df, n=GENERATED_DATA_SIZE):
    """Generate realistic prescription data for Kaiser PAP"""
    patient_ids = patients_df['patient_id'].tolist()
    doctor_ids = doctors_df['doctor_id'].tolist()
    medication_ids = medications_df['medication_id'].tolist()

    data = []
    for i in range(n):
        patient_id = random.choice(patient_ids)
        patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]

        same_state_doctors = doctors_df[doctors_df['state'] == patient['state']]
        if len(same_state_doctors) > 0 and random.random() < 0.8:
            doctor_id = same_state_doctors.sample(1).iloc[0]['doctor_id']
        else:
            doctor_id = random.choice(doctor_ids)

        if patient['is_eligible_for_pap'] and random.random() < 0.7:
            pap_meds = medications_df[medications_df['is_pap_medication']]
            if len(pap_meds) > 0:
                medication = pap_meds.sample(1).iloc[0]
                medication_id = medication['medication_id']
            else:
                medication_id = random.choice(medication_ids)
                medication = medications_df[medications_df['medication_id'] == medication_id].iloc[0]
        else:
            medication_id = random.choice(medication_ids)
            medication = medications_df[medications_df['medication_id'] == medication_id].iloc[0]

        prescription_date = fake.date_between(start_date='-2y', end_date='today')
        max_refills = random.randint(0, 11)
        refills_used = random.randint(0, max_refills)

        if medication['is_pap_medication']:
            quantity = random.randint(1, medication['max_monthly_quantity'])
            days_supply = random.choice([30, 60, 90])
        else:
            quantity = random.randint(1, 30)
            days_supply = random.choice([7, 14, 30])

        data.append({
            'prescription_id': f"RX{10000000 + i}",
            'patient_id': patient_id,
            'doctor_id': doctor_id,
            'medication_id': medication_id,
            'prescription_date': prescription_date,
            'expiration_date': prescription_date + timedelta(days=365),
            'quantity': quantity,
            'days_supply': days_supply,
            'refills_allowed': max_refills,
            'refills_used': refills_used,
            'instructions': f"Take {random.randint(1, 4)} {medication['type'].lower()}s by mouth {random.choice(['daily', 'twice daily', 'as needed'])}",
            'is_pap_prescription': medication['is_pap_medication'],
            'pap_program': 'Kaiser PAP' if medication['is_pap_medication'] else None,
            'status': random.choices(
                ['Active', 'Expired', 'Cancelled', 'Completed'],
                weights=[0.6, 0.2, 0.1, 0.1]
            )[0],
            'fraud_flag': False,
            'notes': fake.sentence() if random.random() < 0.1 else None,
            'created_at': fake.date_between(start_date='-2y', end_date='today')
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    from generator.patient_generator import generate_patients
    from generator.doctor_generator import generate_doctors
    from generator.medication_generator import generate_medications

    patients = generate_patients()
    doctors = generate_doctors()
    medications = generate_medications()
    prescriptions = generate_prescriptions(patients, doctors, medications)
    prescriptions.to_csv(DATA_DIR / "prescriptions.csv", index=False)
    print(f"Generated {len(prescriptions)} prescription records")
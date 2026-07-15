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

def get_diagnosis_description(code):
    descriptions = {
        'C00': 'Malignant neoplasm of lip', 'C01': 'Malignant neoplasm of base of tongue',
        'C02': 'Malignant neoplasm of other parts of tongue', 'C03': 'Malignant neoplasm of gum',
        'C04': 'Malignant neoplasm of floor of mouth', 'C05': 'Malignant neoplasm of palate',
        'C06': 'Malignant neoplasm of other parts of mouth', 'C07': 'Malignant neoplasm of parotid gland',
        'C08': 'Malignant neoplasm of major salivary glands', 'C50': 'Malignant neoplasm of breast',
        'E10': 'Type 1 diabetes mellitus', 'E11': 'Type 2 diabetes mellitus',
        'E13': 'Other specified diabetes mellitus', 'E14': 'Unspecified diabetes mellitus',
        'I10': 'Essential hypertension', 'I11': 'Hypertensive heart disease',
        'I12': 'Hypertensive chronic kidney disease', 'I20': 'Angina pectoris',
        'I21': 'Acute myocardial infarction', 'I25': 'Chronic ischemic heart disease',
        'I48': 'Atrial fibrillation', 'I50': 'Heart failure',
        'J00': 'Acute nasopharyngitis', 'J02': 'Acute pharyngitis',
        'J06': 'Acute upper respiratory infections', 'J20': 'Acute bronchitis',
        'M54': 'Dorsalgia'
    }
    return descriptions.get(code, "Other diagnosis")

def generate_patients(n=GENERATED_DATA_SIZE//5):
    genders = ['M', 'F', 'Other']
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Native American', 'Other']
    ethnicities = ['Non-Hispanic', 'Hispanic']
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    income_levels = ['< $25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '> $100k']
    insurance_types = ['Uninsured', 'Medicaid', 'Medicare', 'Private', 'Underinsured']

    diagnosis_codes = {
        'Cancer': ['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C50'],
        'Diabetes': ['E10', 'E11', 'E13', 'E14'],
        'Cardiovascular': ['I10', 'I11', 'I12', 'I20', 'I21', 'I25', 'I48', 'I50']
    }

    data = []
    for i in range(n):
        first_name = fake.first_name()
        last_name = fake.last_name()
        full_name = f"{first_name} {last_name}"
        ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
        patient_id = f"PAT{1000000 + i}"

        income = random.choice(income_levels)
        insurance = random.choice(insurance_types)
        is_eligible = (income in ['< $25k', '$25k-$50k'] or
                      insurance in ['Uninsured', 'Underinsured'])

        if is_eligible:
            disease = random.choice(list(diagnosis_codes.keys()))
            diagnosis = random.choice(diagnosis_codes[disease])
        else:
            diagnosis = random.choice(['J00', 'J02', 'J06', 'J20', 'M54'])

        data.append({
            'patient_id': patient_id,
            'ssn': ssn,
            'first_name': first_name,
            'last_name': last_name,
            'full_name': full_name,
            'date_of_birth': fake.date_of_birth(minimum_age=18, maximum_age=90),
            'age': random.randint(18, 90),
            'gender': random.choice(genders),
            'race': random.choice(races),
            'ethnicity': random.choice(ethnicities),
            'address': fake.street_address(),
            'city': fake.city(),
            'state': random.choice(states),
            'zip_code': fake.zipcode(),
            'phone': fake.phone_number(),
            'email': f"{first_name.lower()}.{last_name.lower()}@example.com",
            'income_level': income,
            'insurance_type': insurance,
            'insurance_provider': fake.company() if insurance != 'Uninsured' else None,
            'diagnosis_code': diagnosis,
            'diagnosis_description': get_diagnosis_description(diagnosis),
            'is_eligible_for_pap': is_eligible,
            'pap_enrollment_date': fake.date_between(start_date='-2y', end_date='today') if is_eligible else None,
            'fraud_flag': False,
            'created_at': fake.date_between(start_date='-5y', end_date='today')
        })

    df = pd.DataFrame(data)
    eligible_mask = np.random.random(len(df)) < 0.4
    df.loc[eligible_mask, 'is_eligible_for_pap'] = True
    df.loc[eligible_mask, 'income_level'] = random.choices(['< $25k', '$25k-$50k'], k=eligible_mask.sum())
    df.loc[eligible_mask, 'insurance_type'] = random.choices(['Uninsured', 'Underinsured'], k=eligible_mask.sum())

    return df

if __name__ == "__main__":
    patients = generate_patients()
    patients.to_csv(DATA_DIR / "patients.csv", index=False)
    print(f"Generated {len(patients)} patient records")
from faker import Faker
import pandas as pd
import numpy as np
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, PAP_MEDICATIONS

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_medications(n=len(PAP_MEDICATIONS) * 10):
    medication_types = ['Tablet', 'Capsule', 'Injection', 'Topical', 'Oral Solution']
    routes = ['Oral', 'Intravenous', 'Subcutaneous', 'Topical', 'Inhaled']
    frequencies = ['Daily', 'Weekly', 'Biweekly', 'Monthly', 'As needed']

    data = []
    for i, med in enumerate(PAP_MEDICATIONS * (n // len(PAP_MEDICATIONS) + 1)):
        ndc = f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}-{random.randint(10, 99)}"
        data.append({
            'medication_id': f"MED{1000 + i}",
            'name': med,
            'generic_name': f"{med.split('-')[1]} Generic",
            'ndc': ndc,
            'type': random.choice(medication_types),
            'route': random.choice(routes),
            'strength': f"{random.randint(1, 500)}mg",
            'frequency': random.choice(frequencies),
            'manufacturer': f"Kaiser Pharma {random.choice(['Inc', 'LLC', 'Corp'])}",
            'cost_per_unit': round(random.uniform(10, 500), 2),
            'is_pap_medication': True,
            'pap_program': 'Kaiser PAP',
            'requires_prior_auth': random.choice([True, False]),
            'max_monthly_quantity': random.randint(1, 12),
            'side_effects': fake.sentence(),
            'contraindications': fake.sentence(),
            'is_active': True,
            'created_at': fake.date_between(start_date='-5y', end_date='today')
        })

    # Add non-PAP medications
    for i in range(n // 2):
        name = fake.word().capitalize() + str(random.randint(100, 999))
        ndc = f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}-{random.randint(10, 99)}"
        data.append({
            'medication_id': f"MED{1000 + len(PAP_MEDICATIONS) * 10 + i}",
            'name': name,
            'generic_name': f"{name} Generic",
            'ndc': ndc,
            'type': random.choice(medication_types),
            'route': random.choice(routes),
            'strength': f"{random.randint(1, 500)}mg",
            'frequency': random.choice(frequencies),
            'manufacturer': fake.company(),
            'cost_per_unit': round(random.uniform(0.5, 50), 2),
            'is_pap_medication': False,
            'pap_program': None,
            'requires_prior_auth': random.choice([True, False]),
            'max_monthly_quantity': random.randint(1, 30),
            'side_effects': fake.sentence(),
            'contraindications': fake.sentence(),
            'is_active': True,
            'created_at': fake.date_between(start_date='-5y', end_date='today')
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    medications = generate_medications()
    medications.to_csv(DATA_DIR / "medications.csv", index=False)
    print(f"Generated {len(medications)} medication records")
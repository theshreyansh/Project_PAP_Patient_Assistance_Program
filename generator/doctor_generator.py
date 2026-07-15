from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, GENERATED_DATA_SIZE, RANDOM_SEED

# Initialize
fake = Faker()
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_doctors(n=GENERATED_DATA_SIZE//20):
    """Generate realistic doctor data for Kaiser PAP"""
    specialties = [
        'Oncology', 'Cardiology', 'Endocrinology', 'Neurology',
        'Pulmonology', 'Rheumatology', 'Infectious Disease',
        'Hematology', 'Gastroenterology', 'Nephrology'
    ]
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    npi_prefixes = ['1', '2']

    data = []
    for i in range(n):
        first_name = fake.first_name()
        last_name = fake.last_name()
        full_name = f"Dr. {first_name} {last_name}"

        npi = ''.join([random.choice(npi_prefixes)] +
                     [str(random.randint(0, 9)) for _ in range(8)] +
                     [str(random.randint(0, 9))])

        dea = ''.join([random.choice('ABCDEFGHJKLMNPQRSTUVWXY') for _ in range(2)] +
                     [str(random.randint(0, 9)) for _ in range(7)])

        data.append({
            'doctor_id': f"DOC{100000 + i}",
            'npi': npi,
            'dea_number': dea,
            'first_name': first_name,
            'last_name': last_name,
            'full_name': full_name,
            'specialty': random.choice(specialties),
            'hospital_affiliation': fake.company().replace('LLC', 'Hospital').replace('Inc', 'Medical Center'),
            'address': fake.street_address(),
            'city': fake.city(),
            'state': random.choice(states),
            'zip_code': fake.zipcode(),
            'phone': fake.phone_number(),
            'email': f"{first_name.lower()}.{last_name.lower()}@kaiserhealth.org",
            'license_number': f"{random.choice(states)}{random.randint(10000, 99999)}",
            'license_state': random.choice(states),
            'license_expiry': fake.date_between(start_date='+1y', end_date='+5y'),
            'years_of_practice': random.randint(5, 40),
            'is_active': random.choices([True, False], weights=[0.9, 0.1])[0],
            'fraud_flag': False,
            'created_at': fake.date_between(start_date='-5y', end_date='today')
        })

    df = pd.DataFrame(data)
    ca_mask = np.random.random(len(df)) < 0.2
    df.loc[ca_mask, 'state'] = 'CA'
    onco_mask = np.random.random(len(df)) < 0.3
    df.loc[onco_mask, 'specialty'] = 'Oncology'

    return df

if __name__ == "__main__":
    doctors = generate_doctors()
    doctors.to_csv(DATA_DIR / "doctors.csv", index=False)
    print(f"Generated {len(doctors)} doctor records")
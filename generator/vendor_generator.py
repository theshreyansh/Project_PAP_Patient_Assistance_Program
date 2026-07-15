from faker import Faker
import pandas as pd
import numpy as np
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, GENERATED_DATA_SIZE

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_vendors(n=GENERATED_DATA_SIZE//50):
    vendor_types = [
        'Pharmacy', 'Medical Supply', 'DME Supplier',
        'Lab', 'Home Health', 'Transportation'
    ]
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']

    data = []
    for i in range(n):
        name = f"{fake.company()} {random.choice(['Pharmacy', 'Medical', 'Healthcare', 'Supplies'])}"
        data.append({
            'vendor_id': f"VEN{10000 + i}",
            'name': name,
            'type': random.choice(vendor_types),
            'address': fake.street_address(),
            'city': fake.city(),
            'state': random.choice(states),
            'zip_code': fake.zipcode(),
            'phone': fake.phone_number(),
            'email': f"contact@{name.lower().replace(' ', '')}.com",
            'contact_person': fake.name(),
            'tax_id': f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
            'is_pap_vendor': random.choice([True, False]),
            'pap_program': 'Kaiser PAP' if random.choice([True, False]) else None,
            'contract_start_date': fake.date_between(start_date='-5y', end_date='today'),
            'contract_end_date': fake.date_between(start_date='today', end_date='+2y'),
            'is_active': True,
            'fraud_flag': False,
            'created_at': fake.date_between(start_date='-5y', end_date='today')
        })

    df = pd.DataFrame(data)
    pap_mask = np.random.random(len(df)) < 0.3
    df.loc[pap_mask, 'is_pap_vendor'] = True
    df.loc[pap_mask, 'pap_program'] = 'Kaiser PAP'

    return df

if __name__ == "__main__":
    vendors = generate_vendors()
    vendors.to_csv(DATA_DIR / "vendors.csv", index=False)
    print(f"Generated {len(vendors)} vendor records")
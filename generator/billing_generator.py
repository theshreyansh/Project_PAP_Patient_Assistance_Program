from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, GENERATED_DATA_SIZE

# Initialize
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_billing(claims_df, vendors_df, n=GENERATED_DATA_SIZE):
    """Generate realistic billing data for Kaiser PAP"""
    payment_methods = ['Credit Card', 'Check', 'ACH', 'Wire Transfer', 'Insurance']
    billing_statuses = ['Paid', 'Pending', 'Overdue', 'Cancelled', 'Refunded']

    claim_ids = claims_df['claim_id'].tolist()
    vendor_ids = vendors_df['vendor_id'].tolist()

    data = []
    for i in range(n):
        claim = claims_df.sample(1).iloc[0]
        claim_id = claim['claim_id']
        vendor_id = random.choice(vendor_ids)

        billing_date = claim['processed_date'] + timedelta(days=random.randint(1, 14))
        due_date = billing_date + timedelta(days=30)
        amount = claim['total_amount'] * random.uniform(0.9, 1.1)

        data.append({
            'billing_id': f"BIL{10000000 + i}",
            'claim_id': claim_id,
            'vendor_id': vendor_id,
            'billing_date': billing_date,
            'due_date': due_date,
            'amount': round(amount, 2),
            'tax': round(amount * 0.08, 2),
            'total_amount': round(amount * 1.08, 2),
            'payment_method': random.choice(payment_methods),
            'status': random.choices(
                ['Paid', 'Pending', 'Overdue'],
                weights=[0.7, 0.2, 0.1]
            )[0],
            'payment_date': billing_date + timedelta(days=random.randint(1, 30)) if random.random() < 0.7 else None,
            'invoice_number': f"INV{random.randint(1000000, 9999999)}",
            'fraud_flag': False,
            'notes': fake.sentence() if random.random() < 0.1 else None,
            'created_at': fake.date_between(start_date='-2y', end_date='today')
        })

    df = pd.DataFrame(data)

    # Add some realistic patterns
    overdue_mask = np.random.random(len(df)) < 0.05
    df.loc[overdue_mask, 'status'] = 'Overdue'
    df.loc[overdue_mask, 'due_date'] = df.loc[overdue_mask, 'due_date'] - timedelta(days=random.randint(1, 30))

    high_amount_mask = np.random.random(len(df)) < 0.02
    df.loc[high_amount_mask, 'total_amount'] = df.loc[high_amount_mask, 'total_amount'] * random.uniform(2, 5)

    return df

if __name__ == "__main__":
    from generator.claim_generator import generate_claims
    from generator.patient_generator import generate_patients
    from generator.doctor_generator import generate_doctors
    from generator.vendor_generator import generate_vendors

    patients = generate_patients()
    doctors = generate_doctors()
    vendors = generate_vendors()
    claims = generate_claims(patients, doctors)
    billing = generate_billing(claims, vendors)
    billing.to_csv(DATA_DIR / "billing.csv", index=False)
    print(f"Generated {len(billing)} billing records")
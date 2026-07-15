import pandas as pd
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_relationships():
    """Create and validate relationships between all generated data"""
    # Load all data
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    doctors = pd.read_csv(DATA_DIR / "doctors.csv")
    claims = pd.read_csv(DATA_DIR / "claims.csv")
    billing = pd.read_csv(DATA_DIR / "billing.csv")
    eligibility = pd.read_csv(DATA_DIR / "eligibility.csv")
    medications = pd.read_csv(DATA_DIR / "medications.csv")
    vendors = pd.read_csv(DATA_DIR / "vendors.csv")
    prescriptions = pd.read_csv(DATA_DIR / "prescriptions.csv")
    pap = pd.read_csv(DATA_DIR / "pap.csv")

    # Validate and fix relationships
    valid_patient_ids = patients['patient_id'].tolist()
    valid_doctor_ids = doctors['doctor_id'].tolist()
    valid_claim_ids = claims['claim_id'].tolist()
    valid_vendor_ids = vendors['vendor_id'].tolist()
    valid_medication_ids = medications['medication_id'].tolist()

    # Fix claims
    claims['patient_id'] = claims['patient_id'].apply(
        lambda x: x if x in valid_patient_ids else random.choice(valid_patient_ids)
    )
    claims['doctor_id'] = claims['doctor_id'].apply(
        lambda x: x if x in valid_doctor_ids else random.choice(valid_doctor_ids)
    )

    # Fix billing
    billing['claim_id'] = billing['claim_id'].apply(
        lambda x: x if x in valid_claim_ids else random.choice(valid_claim_ids)
    )
    billing['vendor_id'] = billing['vendor_id'].apply(
        lambda x: x if x in valid_vendor_ids else random.choice(valid_vendor_ids)
    )

    # Fix eligibility
    eligibility['patient_id'] = eligibility['patient_id'].apply(
        lambda x: x if x in valid_patient_ids else random.choice(valid_patient_ids)
    )

    # Fix prescriptions
    prescriptions['patient_id'] = prescriptions['patient_id'].apply(
        lambda x: x if x in valid_patient_ids else random.choice(valid_patient_ids)
    )
    prescriptions['doctor_id'] = prescriptions['doctor_id'].apply(
        lambda x: x if x in valid_doctor_ids else random.choice(valid_doctor_ids)
    )
    prescriptions['medication_id'] = prescriptions['medication_id'].apply(
        lambda x: x if x in valid_medication_ids else random.choice(valid_medication_ids)
    )

    # Fix PAP
    pap['patient_id'] = pap['patient_id'].apply(
        lambda x: x if x in valid_patient_ids else random.choice(valid_patient_ids)
    )
    pap['medication_id'] = pap['medication_id'].apply(
        lambda x: x if x in valid_medication_ids else random.choice(valid_medication_ids)
    )

    # Save all data back
    patients.to_csv(DATA_DIR / "patients.csv", index=False)
    doctors.to_csv(DATA_DIR / "doctors.csv", index=False)
    claims.to_csv(DATA_DIR / "claims.csv", index=False)
    billing.to_csv(DATA_DIR / "billing.csv", index=False)
    eligibility.to_csv(DATA_DIR / "eligibility.csv", index=False)
    medications.to_csv(DATA_DIR / "medications.csv", index=False)
    vendors.to_csv(DATA_DIR / "vendors.csv", index=False)
    prescriptions.to_csv(DATA_DIR / "prescriptions.csv", index=False)
    pap.to_csv(DATA_DIR / "pap.csv", index=False)

    print("✓ All relationships validated and fixed")

if __name__ == "__main__":
    create_relationships()
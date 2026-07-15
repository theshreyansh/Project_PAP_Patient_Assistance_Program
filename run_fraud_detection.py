#!/usr/bin/env python3
"""
Run fraud detection on generated healthcare data
"""
import sys
from pathlib import Path
import time
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from fraud.fraud_score import calculate_fraud_scores
from config import DATA_DIR

def main():
    print("="*60)
    print("Kaiser PAP Fraud Detection - Running Fraud Scenarios")
    print("="*60)

    start_time = time.time()

    try:
        # Load all data
        print("\nLoading data...")
        patients = pd.read_csv(DATA_DIR / "patients.csv")
        doctors = pd.read_csv(DATA_DIR / "doctors.csv")
        claims = pd.read_csv(DATA_DIR / "claims.csv")
        billing = pd.read_csv(DATA_DIR / "billing.csv")
        eligibility = pd.read_csv(DATA_DIR / "eligibility.csv")
        medications = pd.read_csv(DATA_DIR / "medications.csv")
        vendors = pd.read_csv(DATA_DIR / "vendors.csv")
        prescriptions = pd.read_csv(DATA_DIR / "prescriptions.csv")
        pap = pd.read_csv(DATA_DIR / "pap.csv")

        print("✓ All data loaded successfully")

        # Run fraud detection
        print("\nRunning fraud detection...")
        results = calculate_fraud_scores(
            claims, patients, doctors, vendors, medications, prescriptions, eligibility, pap, billing
        )

        # Save updated data with fraud flags and scores
        print("\nSaving updated data with fraud information...")
        results['claims'].to_csv(DATA_DIR / "claims_with_fraud.csv", index=False)
        results['patients'].to_csv(DATA_DIR / "patients_with_fraud.csv", index=False)
        results['doctors'].to_csv(DATA_DIR / "doctors_with_fraud.csv", index=False)
        results['vendors'].to_csv(DATA_DIR / "vendors_with_fraud.csv", index=False)

        # Save fraud detection results
        fraud_types = ['duplicate_claims', 'doctor_shopping', 'ghost_patients',
                      'vendor_collusion', 'medication_diversion', 'eligibility_fraud']

        for fraud_type in fraud_types:
            df = results.get(fraud_type, pd.DataFrame())
            if not df.empty:
                df.to_csv(DATA_DIR / f"fraud_{fraud_type}.csv", index=False)
                print(f"✓ Saved {len(df)} {fraud_type} cases")
            else:
                print(f"✓ No {fraud_type} detected")

        # Print summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Fraud Detection Complete!")
        print("="*60)
        print(f"Total time: {elapsed_time:.2f} seconds")

        print("\nFraud Detection Results:")
        for fraud_type in fraud_types:
            df = results.get(fraud_type, pd.DataFrame())
            print(f"- {fraud_type.replace('_', ' ').title()}: {len(df)} cases")

        print(f"\nTotal fraudulent claims: {results['claims']['fraud_flag'].sum()}")
        print(f"Total fraudulent patients: {results['patients']['fraud_flag'].sum()}")
        print(f"Total fraudulent doctors: {results['doctors']['fraud_flag'].sum()}")
        print(f"Total fraudulent vendors: {results['vendors']['fraud_flag'].sum()}")

        print("\nUpdated files saved to data/ directory with '_with_fraud' suffix")
        print("Individual fraud detection results saved as 'fraud_*.csv'")

    except Exception as e:
        print(f"\n❌ Error during fraud detection: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

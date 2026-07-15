import sys
from pathlib import Path
import time

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import all generators
from generator.doctor_generator import generate_doctors
from generator.patient_generator import generate_patients
from generator.medication_generator import generate_medications
from generator.vendor_generator import generate_vendors
from generator.claim_generator import generate_claims
from generator.billing_generator import generate_billing
from generator.eligibility_generator import generate_eligibility
from generator.prescription_generator import generate_prescriptions
from generator.pap_generator import generate_pap_data
from generator.relationships import create_relationships

# Import config
from config import DATA_DIR, GENERATED_DATA_SIZE

def main():
    print("="*60)
    print("Kaiser PAP Fraud Detection - Data Generation")
    print("="*60)
    print(f"Generating {GENERATED_DATA_SIZE:,} total records...")
    print()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Step 1: Generate independent datasets first
        print("1/9: Generating doctors...")
        doctors = generate_doctors()
        doctors.to_csv(DATA_DIR / "doctors.csv", index=False)
        print(f"   ✓ Generated {len(doctors):,} doctor records")

        print("\n2/9: Generating patients...")
        patients = generate_patients()
        patients.to_csv(DATA_DIR / "patients.csv", index=False)
        print(f"   ✓ Generated {len(patients):,} patient records")

        print("\n3/9: Generating medications...")
        medications = generate_medications()
        medications.to_csv(DATA_DIR / "medications.csv", index=False)
        print(f"   ✓ Generated {len(medications):,} medication records")

        print("\n4/9: Generating vendors...")
        vendors = generate_vendors()
        vendors.to_csv(DATA_DIR / "vendors.csv", index=False)
        print(f"   ✓ Generated {len(vendors):,} vendor records")

        # Step 2: Generate datasets that depend on the first set
        print("\n5/9: Generating claims...")
        claims = generate_claims(patients, doctors)
        claims.to_csv(DATA_DIR / "claims.csv", index=False)
        print(f"   ✓ Generated {len(claims):,} claim records")

        print("\n6/9: Generating billing records...")
        billing = generate_billing(claims, vendors)
        billing.to_csv(DATA_DIR / "billing.csv", index=False)
        print(f"   ✓ Generated {len(billing):,} billing records")

        print("\n7/9: Generating eligibility records...")
        eligibility = generate_eligibility(patients)
        eligibility.to_csv(DATA_DIR / "eligibility.csv", index=False)
        print(f"   ✓ Generated {len(eligibility):,} eligibility records")

        print("\n8/9: Generating prescriptions...")
        prescriptions = generate_prescriptions(patients, doctors, medications)
        prescriptions.to_csv(DATA_DIR / "prescriptions.csv", index=False)
        print(f"   ✓ Generated {len(prescriptions):,} prescription records")

        print("\n9/9: Generating PAP records...")
        pap = generate_pap_data(patients, medications)
        pap.to_csv(DATA_DIR / "pap.csv", index=False)
        print(f"   ✓ Generated {len(pap):,} PAP records")

        # Step 3: Create and validate relationships
        print("\nValidating and creating relationships...")
        create_relationships()
        print("   ✓ All relationships validated and created")

        # Final summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("Data Generation Complete!")
        print("="*60)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Data saved to: {DATA_DIR.absolute()}")
        print("\nGenerated files:")
        for file in DATA_DIR.glob("*.csv"):
            size = file.stat().st_size / 1024  # KB
            print(f"  - {file.name}: {size:.1f} KB")

    except Exception as e:
        print(f"\n❌ Error during data generation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
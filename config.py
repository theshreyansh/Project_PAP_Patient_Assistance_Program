import os
from pathlib import Path

# Project configuration
PROJECT_NAME = "Kaiser PAP Fraud Detection"
PROJECT_VERSION = "1.0.0"
COMPANY_NAME = "Kaiser Pharmaceuticals"
PAP_PROGRAM_NAME = "Kaiser Patient Assistance Program"

# Data configuration
DATA_DIR = Path("data")
GENERATED_DATA_SIZE = 100000  # Number of records to generate
RANDOM_SEED = 42

# Fraud configuration
FRAUD_RATIO = 0.05  # 5% of records will be fraudulent
FRAUD_SCENARIOS = [
    "duplicate_claims",
    "doctor_shopping",
    "ghost_patients",
    "vendor_collusion",
    "medication_diversion",
    "eligibility_fraud"
]

# UI configuration
THEME_COLOR = "#2E86C1"
SECONDARY_COLOR = "#1A5276"
BACKGROUND_COLOR = "#F5F7FA"
TEXT_COLOR = "#2C3E50"
CARD_COLOR = "#FFFFFF"

# File paths
LOGO_PATH = "assets/logo.png"
HEALTHCARE_GIF = "assets/healthcare.gif"
FRAUD_GIF = "assets/fraud.gif"

# Data files
DATA_FILES = {
    "doctors": "doctors.csv",
    "patients": "patients.csv",
    "claims": "claims.csv",
    "billing": "billing.csv",
    "eligibility": "eligibility.csv",
    "medications": "medications.csv",
    "vendors": "vendors.csv",
    "prescriptions": "prescriptions.csv",
    "pap": "pap.csv"  # Patient Assistance Program data
}

# PAP specific configurations
PAP_MEDICATIONS = [
    "Kaiser-OncoX", "Kaiser-Diabeta", "Kaiser-Cardio",
    "Kaiser-Respira", "Kaiser-Neurolin", "Kaiser-Immuno"
]
PAP_ELIGIBILITY_CRITERIA = {
    "income_threshold": 50000,  # Annual income in USD
    "insurance_status": ["uninsured", "underinsured"],
    "diagnosis_codes": ["C00-C97", "E10-E14", "I00-I99"]  # Cancer, Diabetes, Cardiovascular
}

# Fraud configuration (add to existing config.py)
FRAUD_THRESHOLDS = {
    'duplicate_claims': 0.95,      # Similarity threshold for duplicate claims
    'doctor_shopping': 3,          # More than 3 doctors in 30 days
    'ghost_patients': 0.8,         # Similarity score for ghost patients
    'vendor_collusion': 0.5,       # Jaccard similarity for vendor collusion
    'medication_diversion': 1.5,   # Z-score threshold for medication diversion
    'patient_fraud': 0.5,          # Fraud score threshold for patients
    'doctor_fraud': 0.5,           # Fraud score threshold for doctors
    'vendor_fraud': 0.5,           # Fraud score threshold for vendors
    'eligibility_fraud': 0.8       # Fraud score threshold for eligibility
}
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_DIR, FRAUD_SCENARIOS
from generator.claim_generator import generate_claims
from generator.doctor_generator import generate_doctors
from generator.eligibility_generator import generate_eligibility
from generator.medication_generator import generate_medications
from generator.pap_generator import generate_pap_data
from generator.patient_generator import generate_patients
from generator.prescription_generator import generate_prescriptions
from generator.relationships import create_relationships
from generator.vendor_generator import generate_vendors
from fraud.fraud_score import calculate_fraud_scores


APP_TITLE = "Pharma Data Due Diligence Cockpit"
APP_SUBTITLE = "Executive-ready risk, control, and due diligence intelligence for pharma PAP and claims data"

DATE_COLUMNS = {
    "patients": ["date_of_birth", "pap_enrollment_date", "created_at"],
    "doctors": ["license_expiry", "created_at"],
    "claims": ["claim_date", "processed_date", "created_at"],
    "billing": ["billing_date", "due_date", "payment_date", "created_at"],
    "eligibility": ["application_date", "expiration_date", "review_date", "created_at"],
    "prescriptions": ["prescription_date", "expiration_date", "created_at"],
    "vendors": ["contract_start_date", "contract_end_date", "created_at"],
    "pap": ["created_at"],
    "medications": ["created_at"],
}


def inject_css() -> None:
    css_path = Path(__file__).with_name("style.css")
    base_css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def masked_ssn(value: object) -> str:
    text = str(value)
    if "-" in text and len(text) >= 4:
        return f"***-**-{text[-4:]}"
    if len(text) >= 4:
        return f"***{text[-4:]}"
    return "***"


def data_signature() -> Tuple[Tuple[str, int, int], ...]:
    files = {
        "patients": "patients.csv",
        "doctors": "doctors.csv",
        "claims": "claims.csv",
        "billing": "billing.csv",
        "eligibility": "eligibility.csv",
        "medications": "medications.csv",
        "vendors": "vendors.csv",
        "prescriptions": "prescriptions.csv",
        "pap": "pap.csv",
    }
    signature: List[Tuple[str, int, int]] = []
    for key, filename in sorted(files.items()):
        path = DATA_DIR / filename
        if path.exists():
            stat = path.stat()
            signature.append((key, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            signature.append((key, -1, -1))
    return tuple(signature)


def read_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for column in DATE_COLUMNS.get(name, []):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data(signature: Tuple[Tuple[str, int, int], ...]) -> Dict[str, pd.DataFrame]:
    return {name: read_table(name) for name in [
        "patients",
        "doctors",
        "claims",
        "billing",
        "eligibility",
        "medications",
        "vendors",
        "prescriptions",
        "pap",
    ]}


def subset_for_review(data: Dict[str, pd.DataFrame], claim_limit: int) -> Dict[str, pd.DataFrame]:
    claims = data["claims"].copy()
    if not claims.empty and "claim_date" in claims.columns:
        claims = claims.sort_values("claim_date", ascending=False)
    if claim_limit < len(claims):
        claims = claims.head(claim_limit).copy()

    patient_ids = set(claims.get("patient_id", pd.Series(dtype=str)).dropna().astype(str))
    doctor_ids = set(claims.get("doctor_id", pd.Series(dtype=str)).dropna().astype(str))

    billing = data["billing"]
    if not billing.empty and "claim_id" in billing.columns:
        billing = billing[billing["claim_id"].isin(claims["claim_id"])].copy()
    vendor_ids = set(billing.get("vendor_id", pd.Series(dtype=str)).dropna().astype(str))

    patients = data["patients"]
    if not patients.empty and patient_ids:
        patients = patients[patients["patient_id"].astype(str).isin(patient_ids)].copy()

    doctors = data["doctors"]
    if not doctors.empty and doctor_ids:
        doctors = doctors[doctors["doctor_id"].astype(str).isin(doctor_ids)].copy()

    eligibility = data["eligibility"]
    if not eligibility.empty and patient_ids:
        eligibility = eligibility[eligibility["patient_id"].astype(str).isin(patient_ids)].copy()

    prescriptions = data["prescriptions"]
    if not prescriptions.empty:
        mask = prescriptions["patient_id"].astype(str).isin(patient_ids) | prescriptions["doctor_id"].astype(str).isin(doctor_ids)
        prescriptions = prescriptions[mask].copy()

    vendors = data["vendors"]
    if not vendors.empty and vendor_ids:
        vendors = vendors[vendors["vendor_id"].astype(str).isin(vendor_ids)].copy()

    pap = data["pap"]
    if not pap.empty and "patient_id" in pap.columns and patient_ids:
        pap = pap[pap["patient_id"].astype(str).isin(patient_ids)].copy()

    return {
        "patients": patients.reset_index(drop=True),
        "doctors": doctors.reset_index(drop=True),
        "claims": claims.reset_index(drop=True),
        "billing": billing.reset_index(drop=True),
        "eligibility": eligibility.reset_index(drop=True),
        "medications": data["medications"].reset_index(drop=True),
        "vendors": vendors.reset_index(drop=True),
        "prescriptions": prescriptions.reset_index(drop=True),
        "pap": pap.reset_index(drop=True),
    }


def run_generation_with_status() -> None:
    with st.status("Preparing enterprise dataset...", expanded=True) as status:
        status.write("Generating doctors...")
        doctors = generate_doctors()
        doctors.to_csv(DATA_DIR / "doctors.csv", index=False)

        status.write("Generating patients...")
        patients = generate_patients()
        patients.to_csv(DATA_DIR / "patients.csv", index=False)

        status.write("Generating medications...")
        medications = generate_medications()
        medications.to_csv(DATA_DIR / "medications.csv", index=False)

        status.write("Generating vendors...")
        vendors = generate_vendors()
        vendors.to_csv(DATA_DIR / "vendors.csv", index=False)

        status.write("Generating claims...")
        claims = generate_claims(patients, doctors)
        claims.to_csv(DATA_DIR / "claims.csv", index=False)

        status.write("Generating billing records...")
        from generator.billing_generator import generate_billing

        billing = generate_billing(claims, vendors)
        billing.to_csv(DATA_DIR / "billing.csv", index=False)

        status.write("Generating eligibility files...")
        eligibility = generate_eligibility(patients)
        eligibility.to_csv(DATA_DIR / "eligibility.csv", index=False)

        status.write("Generating prescriptions...")
        prescriptions = generate_prescriptions(patients, doctors, medications)
        prescriptions.to_csv(DATA_DIR / "prescriptions.csv", index=False)

        status.write("Generating PAP data...")
        pap = generate_pap_data(patients, medications)
        pap.to_csv(DATA_DIR / "pap.csv", index=False)

        status.write("Finalizing relationships and validation checks...")
        create_relationships()
        status.update(label="Enterprise dataset refreshed", state="complete", expanded=False)


def run_fraud_with_status(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    with st.status("Executing due diligence checks...", expanded=True) as status:
        status.write("Validating claims integrity...")
        results = calculate_fraud_scores(
            data["claims"],
            data["patients"],
            data["doctors"],
            data["vendors"],
            data["medications"],
            data["prescriptions"],
            data["eligibility"],
            data["pap"],
            data["billing"],
            progress_callback=lambda message: status.write(message),
        )
        status.update(label="Due diligence checks completed", state="complete", expanded=False)
        return results


def run_quality_checks(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    claims = data["claims"]
    patients = data["patients"]
    doctors = data["doctors"]
    billing = data["billing"]
    eligibility = data["eligibility"]

    rows: List[Dict[str, object]] = []

    def add_issue(check: str, severity: str, count: int, total: int, owner: str, implication: str) -> None:
        rows.append({
            "check": check,
            "severity": severity,
            "count": int(count),
            "rate_pct": round((count / total * 100), 2) if total else 0.0,
            "owner": owner,
            "implication": implication,
        })

    if not claims.empty:
        add_issue(
            "Claims with missing patient or doctor links",
            "High",
            int((claims["patient_id"].isna() | claims["doctor_id"].isna()).sum()),
            len(claims),
            "Data Steward",
            "Investigate upstream intake and master-data matching.",
        )
        if {"processed_date", "claim_date"}.issubset(claims.columns):
            add_issue(
                "Claims processed before claim date",
                "High",
                int((claims["processed_date"] < claims["claim_date"]).sum()),
                len(claims),
                "Operations",
                "Review adjudication timing and interface timestamps.",
            )
        claim_keys = [c for c in ["patient_id", "doctor_id", "medication", "claim_date", "total_amount"] if c in claims.columns]
        if claim_keys:
            add_issue(
                "Duplicate claim patterns",
                "High",
                int(claims.duplicated(subset=claim_keys).sum()),
                len(claims),
                "Revenue Assurance",
                "Add dedupe controls at intake and pre-adjudication.",
            )
        if "total_amount" in claims.columns:
            add_issue(
                "Claims in top 1% value band",
                "Medium",
                int((claims["total_amount"] > claims["total_amount"].quantile(0.99)).sum()),
                len(claims),
                "Finance",
                "Introduce exception review for unusually high claim value.",
            )

    if not patients.empty:
        if "ssn" in patients.columns:
            add_issue(
                "Duplicate SSN records",
                "High",
                int(patients["ssn"].duplicated().sum()),
                len(patients),
                "Master Data",
                "Strengthen identity verification and dedupe at enrollment.",
            )
        if "diagnosis_code" in patients.columns:
            add_issue(
                "Patients missing diagnosis code",
                "Medium",
                int(patients["diagnosis_code"].isna().sum()),
                len(patients),
                "Clinical Ops",
                "Require diagnosis completion before PAP approval.",
            )

    if not billing.empty:
        add_issue(
            "Billing records without matching claim",
            "High",
            int((~billing["claim_id"].isin(claims["claim_id"])).sum()),
            len(billing),
            "AP / Billing",
            "Reconcile vendor billing back to adjudicated claims.",
        )

    if not eligibility.empty:
        add_issue(
            "Eligibility files with repeated applications",
            "Medium",
            int(eligibility["patient_id"].duplicated().sum()),
            len(eligibility),
            "Case Management",
            "Review repeat applications and documentation quality.",
        )

    if not doctors.empty and "license_expiry" in doctors.columns:
        add_issue(
            "Doctors with expired licenses",
            "High",
            int((doctors["license_expiry"] < pd.Timestamp.today()).sum()),
            len(doctors),
            "Provider Ops",
            "Flag provider access if credentialing is stale.",
        )

    return pd.DataFrame(rows)


def business_recommendations(results: Dict[str, pd.DataFrame], quality: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    scenario_actions = [
        ("duplicate_claims", "Billing", "Add duplicate detection before payment release and reject identical resubmissions."),
        ("doctor_shopping", "Clinical Ops", "Trigger utilization review for members who cycle across multiple prescribers."),
        ("ghost_patients", "Master Data", "Re-verify identity artifacts and normalize name/address matching at enrollment."),
        ("vendor_collusion", "Finance / Procurement", "Review vendor concentration, cross-claim overlap, and contracting exceptions."),
        ("medication_diversion", "Pharmacy", "Tighten PAP quantity limits and prescriber monitoring for high-risk therapies."),
        ("eligibility_fraud", "Case Management", "Re-certify income and insurance documentation at renewal and exceptions."),
    ]

    rows: List[Dict[str, object]] = []
    for scenario, owner, action in scenario_actions:
        df = results.get(scenario, pd.DataFrame())
        if df.empty:
            continue
        exposure = 0.0
        if "claim_id" in df.columns and "claim_id" in claims.columns and "total_amount" in claims.columns:
            exposure = float(claims[claims["claim_id"].isin(df["claim_id"])]["total_amount"].fillna(0).sum())
        rows.append({
            "priority": len(df),
            "theme": scenario.replace("_", " ").title(),
            "owner": owner,
            "finding_count": len(df),
            "estimated_exposure": exposure,
            "recommended_action": action,
            "why_it_matters": "Higher exposure, brand risk, and control breakdown signals for the executive team.",
        })

    if not quality.empty:
        major_quality = quality[quality["count"] > 0].sort_values("count", ascending=False).head(3)
        for _, row in major_quality.iterrows():
            rows.append({
                "priority": int(row["count"]),
                "theme": f"Data Quality: {row['check']}",
                "owner": row["owner"],
                "finding_count": int(row["count"]),
                "estimated_exposure": 0.0,
                "recommended_action": row["implication"],
                "why_it_matters": "Executive reporting depends on reliable data lineage and controls.",
            })

    if not rows:
        return pd.DataFrame(columns=[
            "priority",
            "theme",
            "owner",
            "finding_count",
            "estimated_exposure",
            "recommended_action",
            "why_it_matters",
        ])

    return pd.DataFrame(rows).sort_values(["priority", "estimated_exposure"], ascending=[False, False]).reset_index(drop=True)


def scenario_summary(results: Dict[str, pd.DataFrame], claims: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in FRAUD_SCENARIOS:
        df = results.get(scenario, pd.DataFrame())
        exposure = 0.0
        if not df.empty and "claim_id" in df.columns and "total_amount" in claims.columns:
            exposure = float(claims[claims["claim_id"].isin(df["claim_id"])]["total_amount"].fillna(0).sum())
        rows.append({
            "scenario": scenario.replace("_", " ").title(),
            "count": int(len(df)),
            "estimated_exposure": exposure,
            "max_score": round(float(df["score"].max()), 3) if not df.empty and "score" in df.columns else 0.0,
        })
    return pd.DataFrame(rows)


def display_hero(total_claims: int, flagged_claims: int, exposure: float, quality_issues: int, analysis_seconds: float) -> None:
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="eyebrow">Pharma diligence platform</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Claims reviewed", f"{total_claims:,}")
    c2.metric("Flagged claims", f"{flagged_claims:,}")
    c3.metric("Flag rate", f"{(flagged_claims / total_claims * 100):.1f}%" if total_claims else "0.0%")
    c4.metric("Estimated exposure", format_currency(exposure))
    c5.metric("Run time", f"{analysis_seconds:.1f}s")

    if quality_issues > 0:
        st.warning(f"{quality_issues} data-quality issues were surfaced and mapped to control actions.")
    else:
        st.success("No material data-quality breaks were detected in the current analysis sample.")


def build_figures(summary: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    fig1 = px.bar(
        summary,
        x="scenario",
        y="count",
        color="estimated_exposure",
        text="count",
        color_continuous_scale=["#CFE8FF", "#0C4A6E"],
        title="Findings by scenario",
    )
    fig1.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="", yaxis_title="Flagged records")

    fig2 = px.pie(
        summary[summary["count"] > 0],
        values="count",
        names="scenario",
        title="Finding mix",
        hole=0.45,
    )
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig1, fig2


def claims_view_table(claims: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    cols = [c for c in ["claim_id", "patient_id", "doctor_id", "claim_type", "claim_date", "total_amount", "fraud_score", "fraud_flag"] if c in claims.columns]
    view = claims.loc[:, cols].copy()
    if "fraud_score" in view.columns:
        view["fraud_score"] = view["fraud_score"].round(3)
    if "total_amount" in view.columns:
        view["total_amount"] = view["total_amount"].round(2)
    return view.sort_values(by=[c for c in ["fraud_score", "total_amount"] if c in view.columns], ascending=False).head(limit)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    inject_css()

    st.sidebar.title("Control Center")
    st.sidebar.caption("Tune the analysis for an executive walkthrough.")

    data = load_data(data_signature())
    missing_files = [name for name, df in data.items() if df.empty]
    if missing_files:
        st.error(f"Missing required data files for: {', '.join(missing_files)}")
        st.info("Run `python generate_data.py` to rebuild the dataset, then refresh the app.")
        return

    mode = st.sidebar.selectbox(
        "Analysis depth",
        ["Executive snapshot - 2,500 claims", "Balanced review - 10,000 claims", "Full dataset"],
        index=0,
    )
    claim_limit = 2500 if "2,500" in mode else 10000 if "10,000" in mode else len(data["claims"])

    if st.sidebar.button("Re-run analysis"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("Refresh synthetic dataset"):
        run_generation_with_status()
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### What this platform shows")
    st.sidebar.write("• Due diligence checks on claims, patient, provider, vendor, and eligibility data")
    st.sidebar.write("• Fraud test results with business impact")
    st.sidebar.write("• Executive actions for CXO review")

    quality = run_quality_checks(data)
    subset = subset_for_review(data, claim_limit)

    start = time.perf_counter()
    results = run_fraud_with_status(subset)
    elapsed = time.perf_counter() - start

    claims = results["claims"]
    summary = scenario_summary(results, claims)
    recommendations = business_recommendations(results, quality, claims)
    flagged_claims = int(claims["fraud_flag"].sum()) if "fraud_flag" in claims.columns else 0
    exposure = float(claims.loc[claims["fraud_flag"] == True, "total_amount"].fillna(0).sum()) if "total_amount" in claims.columns else 0.0

    display_hero(len(claims), flagged_claims, exposure, int((quality["count"] > 0).sum()) if not quality.empty else 0, elapsed)

    tab_overview, tab_quality, tab_findings, tab_actions, tab_drilldown, tab_export = st.tabs(
        ["Overview", "Data Quality", "Fraud Tests", "CXO Actions", "Drilldown", "Export"]
    )

    with tab_overview:
        fig1, fig2 = build_figures(summary)
        left, right = st.columns(2)
        left.plotly_chart(fig1, use_container_width=True)
        right.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Executive summary")
        top = summary.sort_values("count", ascending=False).head(3)
        if top.empty or top["count"].sum() == 0:
            st.info("No fraud cases were identified in the current sample, which is itself a useful control signal.")
        else:
            for _, row in top.iterrows():
                st.markdown(f"- **{row['scenario']}**: {int(row['count']):,} cases, estimated exposure {format_currency(row['estimated_exposure'])}")

    with tab_quality:
        st.markdown("#### Data quality control tower")
        if quality.empty:
            st.info("No quality issues detected.")
        else:
            st.dataframe(quality.sort_values(["severity", "count"], ascending=[True, False]), use_container_width=True)

        st.markdown("#### High-risk examples")
        q1, q2 = st.columns(2)
        if not data["patients"].empty and "ssn" in data["patients"].columns:
            duplicate_ssn = data["patients"][data["patients"]["ssn"].duplicated(keep=False)].copy()
            if not duplicate_ssn.empty:
                duplicate_ssn["ssn"] = duplicate_ssn["ssn"].map(masked_ssn)
            q1.write("Duplicate identity records")
            q1.dataframe(duplicate_ssn.head(10), use_container_width=True)
        if not data["billing"].empty:
            overdue = data["billing"][data["billing"]["status"].astype(str).str.contains("Overdue", case=False, na=False)]
            q2.write("Overdue billing records")
            q2.dataframe(overdue.head(10), use_container_width=True)

    with tab_findings:
        st.markdown("#### Scenario outcomes")
        st.dataframe(summary, use_container_width=True)
        scenario_choice = st.selectbox("Choose scenario", [scenario.replace("_", " ").title() for scenario in FRAUD_SCENARIOS])
        scenario_key = scenario_choice.lower().replace(" ", "_")
        scenario_df = results.get(scenario_key, pd.DataFrame())
        if scenario_df.empty:
            st.info("No cases returned for this scenario in the current sample.")
        else:
            preview = scenario_df.copy()
            if "patient_id" in preview.columns:
                preview["patient_id"] = preview["patient_id"].astype(str)
            st.dataframe(preview.head(50), use_container_width=True)

    with tab_actions:
        st.markdown("#### Prioritized action plan")
        if recommendations.empty:
            st.success("No major action items in the current sample. Continue monitoring and trending.")
        else:
            st.dataframe(
                recommendations[["theme", "owner", "finding_count", "estimated_exposure", "recommended_action"]],
                use_container_width=True,
            )
            for _, rec in recommendations.head(5).iterrows():
                st.markdown(
                    f"""
**{rec['theme']}**

Owner: {rec['owner']}

Action: {rec['recommended_action']}

Why it matters: {rec['why_it_matters']}
"""
                )

    with tab_drilldown:
        drill_choice = st.selectbox("Entity view", ["Claims", "Patients", "Doctors", "Vendors"], index=0)
        if drill_choice == "Claims":
            st.dataframe(claims_view_table(claims), use_container_width=True)
        elif drill_choice == "Patients":
            cols = [c for c in ["patient_id", "full_name", "state", "income_level", "insurance_type", "fraud_score", "fraud_flag"] if c in results["patients"].columns]
            st.dataframe(results["patients"][cols].sort_values("fraud_score", ascending=False).head(50), use_container_width=True)
        elif drill_choice == "Doctors":
            cols = [c for c in ["doctor_id", "full_name", "specialty", "state", "fraud_score", "fraud_flag"] if c in results["doctors"].columns]
            st.dataframe(results["doctors"][cols].sort_values("fraud_score", ascending=False).head(50), use_container_width=True)
        else:
            cols = [c for c in ["vendor_id", "name", "type", "state", "fraud_score", "fraud_flag"] if c in results["vendors"].columns]
            st.dataframe(results["vendors"][cols].sort_values("fraud_score", ascending=False).head(50), use_container_width=True)

    with tab_export:
        st.markdown("#### Download executive outputs")
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Download scenario summary",
            summary.to_csv(index=False).encode("utf-8"),
            file_name="scenario_summary.csv",
            mime="text/csv",
        )
        col2.download_button(
            "Download data quality checks",
            quality.to_csv(index=False).encode("utf-8"),
            file_name="data_quality_checks.csv",
            mime="text/csv",
        )
        col3.download_button(
            "Download recommendations",
            recommendations.to_csv(index=False).encode("utf-8"),
            file_name="cxo_actions.csv",
            mime="text/csv",
        )

        report = pd.DataFrame({
            "metric": ["claims_reviewed", "flagged_claims", "estimated_exposure", "quality_issues"],
            "value": [len(claims), flagged_claims, exposure, int((quality["count"] > 0).sum()) if not quality.empty else 0],
        })
        st.download_button(
            "Download executive snapshot",
            report.to_csv(index=False).encode("utf-8"),
            file_name="executive_snapshot.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()

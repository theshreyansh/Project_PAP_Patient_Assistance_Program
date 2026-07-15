from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import time

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import DATA_DIR, FRAUD_SCENARIOS
from fraud.fraud_score import calculate_fraud_scores


APP_TITLE = "Pharma Data Due Diligence Cockpit"
APP_SUBTITLE = "Executive-grade risk, control, and due diligence intelligence for pharma PAP and claims operations"

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
    st.markdown(
        f"""
        <style>
        {base_css}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        div[data-testid="collapsedControl"] {{
            display: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def masked_id(value: object) -> str:
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"X{int(digest[:6], 16) % 10000:04d}"


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


def run_generation_with_status() -> None:
    from generator.billing_generator import generate_billing
    from generator.claim_generator import generate_claims
    from generator.doctor_generator import generate_doctors
    from generator.eligibility_generator import generate_eligibility
    from generator.medication_generator import generate_medications
    from generator.pap_generator import generate_pap_data
    from generator.patient_generator import generate_patients
    from generator.prescription_generator import generate_prescriptions
    from generator.relationships import create_relationships
    from generator.vendor_generator import generate_vendors

    with st.status("Refreshing enterprise dataset", expanded=True) as status:
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
    with st.status("Running due diligence analytics", expanded=True) as status:
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
        status.update(label="Analytics complete", state="complete", expanded=False)
        return results


def cached_analysis(signature: Tuple[Tuple[str, int, int], ...]) -> Dict[str, object]:
    data = load_data(signature)
    quality = run_quality_checks(data)
    start = time.perf_counter()
    results = run_fraud_with_status(data)
    elapsed = time.perf_counter() - start

    return {
        "data": data,
        "results": results,
        "quality": quality,
        "summary": build_scenario_summary(results, results["claims"]),
        "recommendations": build_recommendations(results, quality, results["claims"]),
        "charts": build_dashboard_charts(data, results),
        "elapsed": elapsed,
    }


def ensure_analysis(signature: Tuple[Tuple[str, int, int], ...]) -> Dict[str, object]:
    cache_key = signature
    if st.session_state.get("analysis_cache_key") == cache_key:
        return st.session_state["analysis_cache"]

    artifact = cached_analysis(signature)
    st.session_state["analysis_cache_key"] = cache_key
    st.session_state["analysis_cache"] = artifact
    return artifact


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
        add_issue("Claims with missing patient or doctor links", "High",
                  int((claims["patient_id"].isna() | claims["doctor_id"].isna()).sum()),
                  len(claims), "Data Steward", "Investigate upstream intake and master-data matching.")
        if {"processed_date", "claim_date"}.issubset(claims.columns):
            add_issue("Claims processed before claim date", "High",
                      int((claims["processed_date"] < claims["claim_date"]).sum()),
                      len(claims), "Operations", "Review adjudication timing and interface timestamps.")
        claim_keys = [c for c in ["patient_id", "doctor_id", "medication", "claim_date", "total_amount"] if c in claims.columns]
        if claim_keys:
            add_issue("Duplicate claim patterns", "High",
                      int(claims.duplicated(subset=claim_keys).sum()),
                      len(claims), "Revenue Assurance", "Add dedupe controls at intake and pre-adjudication.")
        if "total_amount" in claims.columns:
            add_issue("Claims in top 1% value band", "Medium",
                      int((claims["total_amount"] > claims["total_amount"].quantile(0.99)).sum()),
                      len(claims), "Finance", "Introduce exception review for unusually high claim value.")

    if not patients.empty and "ssn" in patients.columns:
        add_issue("Duplicate SSN records", "High",
                  int(patients["ssn"].duplicated().sum()), len(patients),
                  "Master Data", "Strengthen identity verification and dedupe at enrollment.")
    if not patients.empty and "diagnosis_code" in patients.columns:
        add_issue("Patients missing diagnosis code", "Medium",
                  int(patients["diagnosis_code"].isna().sum()), len(patients),
                  "Clinical Ops", "Require diagnosis completion before PAP approval.")
    if not billing.empty:
        add_issue("Billing records without matching claim", "High",
                  int((~billing["claim_id"].isin(claims["claim_id"])).sum()), len(billing),
                  "AP / Billing", "Reconcile vendor billing back to adjudicated claims.")
    if not eligibility.empty:
        add_issue("Eligibility files with repeated applications", "Medium",
                  int(eligibility["patient_id"].duplicated().sum()), len(eligibility),
                  "Case Management", "Review repeat applications and documentation quality.")
    if not doctors.empty and "license_expiry" in doctors.columns:
        add_issue("Doctors with expired licenses", "High",
                  int((doctors["license_expiry"] < pd.Timestamp.today()).sum()), len(doctors),
                  "Provider Ops", "Flag provider access if credentialing is stale.")

    return pd.DataFrame(rows)


def build_recommendations(results: Dict[str, pd.DataFrame], quality: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    actions = {
        "duplicate_claims": ("Billing", "Add duplicate detection before payment release and reject identical resubmissions."),
        "doctor_shopping": ("Clinical Ops", "Trigger utilization review for members who cycle across multiple prescribers."),
        "ghost_patients": ("Master Data", "Re-verify identity artifacts and normalize name/address matching at enrollment."),
        "vendor_collusion": ("Finance / Procurement", "Review vendor concentration, cross-claim overlap, and contracting exceptions."),
        "medication_diversion": ("Pharmacy", "Tighten PAP quantity limits and prescriber monitoring for high-risk therapies."),
        "eligibility_fraud": ("Case Management", "Re-certify income and insurance documentation at renewal and exceptions."),
    }
    for scenario, (owner, action) in actions.items():
        df = results.get(scenario, pd.DataFrame())
        if df.empty:
            continue
        exposure = 0.0
        if "claim_id" in df.columns and "total_amount" in claims.columns:
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
        top_quality = quality[quality["count"] > 0].sort_values("count", ascending=False).head(4)
        for _, row in top_quality.iterrows():
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
        return pd.DataFrame(columns=["priority", "theme", "owner", "finding_count", "estimated_exposure", "recommended_action", "why_it_matters"])
    return pd.DataFrame(rows).sort_values(["priority", "estimated_exposure"], ascending=[False, False]).reset_index(drop=True)


def build_scenario_summary(results: Dict[str, pd.DataFrame], claims: pd.DataFrame) -> pd.DataFrame:
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


def provider_type_for_doctor(row: pd.Series) -> str:
    specialty = str(row.get("specialty", "")).lower()
    affiliation = str(row.get("hospital_affiliation", "")).lower()
    if "hospital" in affiliation or "medical center" in affiliation:
        return "Hospital"
    if specialty in {"oncology", "cardiology", "endocrinology", "hematology"}:
        return "Clinic"
    return "Individual"


def build_dashboard_charts(data: Dict[str, pd.DataFrame], results: Dict[str, pd.DataFrame]) -> Dict[str, go.Figure]:
    claims = data["claims"].copy()
    doctors = data["doctors"].copy()
    patients = data["patients"].copy()
    prescriptions = data["prescriptions"].copy()
    medications = data["medications"].copy()
    vendors = data["vendors"].copy()
    billing = data["billing"].copy()

    charts: Dict[str, go.Figure] = {}

    if not doctors.empty:
        doctors["provider_type"] = doctors.apply(provider_type_for_doctor, axis=1)
        completed = (doctors["is_active"].fillna(False) & (doctors["license_expiry"] > pd.Timestamp.today())).astype(int)
        chart_df = doctors.groupby("provider_type").agg(
            total=("doctor_id", "count"),
            completed=("doctor_id", lambda s: int(completed.loc[s.index].sum()))
        ).reset_index()
        chart_df["pct_completed"] = chart_df["completed"] / chart_df["total"] * 100
        charts["due_diligence"] = px.bar(
            chart_df, x="provider_type", y="pct_completed", text=chart_df["pct_completed"].round(1),
            title="Due Diligence Completion by Provider Type",
            labels={"provider_type": "Provider Type", "pct_completed": "% Completed Due Diligence"},
            color="pct_completed", color_continuous_scale=["#dbeafe", "#0f766e"]
        )
        charts["due_diligence"].update_layout(height=420, yaxis_range=[0, 100], coloraxis_showscale=False)
        if not chart_df.empty:
            worst = chart_df.sort_values("pct_completed").iloc[0]
            charts["due_diligence"].add_annotation(
                x=0.5, y=1.08, xref="paper", yref="paper",
                text=f"Only {worst['pct_completed']:.0f}% of {worst['provider_type'].lower()} providers have completed due diligence",
                showarrow=False, font=dict(size=12, color="#0f3550")
            )

    if not claims.empty:
        claims = claims.copy()
        claims["month"] = claims["claim_date"].dt.to_period("M").dt.to_timestamp()
        claims["status_bucket"] = "Valid"
        claims.loc[claims["status"].astype(str).str.contains("Rejected", case=False, na=False), "status_bucket"] = "Rejected"
        if "fraud_flag" in claims.columns:
            claims.loc[claims["fraud_flag"].fillna(False), "status_bucket"] = "Flagged"
        integrity = claims.groupby(["month", "status_bucket"]).size().reset_index(name="count")
        charts["claims_integrity"] = px.bar(
            integrity, x="month", y="count", color="status_bucket", barmode="stack",
            title="Claims Integrity Validation",
            labels={"month": "Month", "count": "# of Claims", "status_bucket": "Status"}
        )
        charts["claims_integrity"].update_layout(height=420, legend_title_text="")

        duplicates = results.get("duplicate_claims", pd.DataFrame())
        dup_series = pd.DataFrame({"date": []})
        if not duplicates.empty:
            dup_series = claims[claims["claim_id"].isin(duplicates["claim_id"])].copy()
            dup_series = dup_series.groupby(dup_series["claim_date"].dt.to_period("M").dt.to_timestamp()).size().reset_index(name="count")
            dup_series.columns = ["date", "count"]
        charts["duplicates"] = px.line(
            dup_series if not dup_series.empty else pd.DataFrame({"date": [], "count": []}),
            x="date", y="count", markers=True, title="Duplicate Claims Detection"
        )
        charts["duplicates"].update_layout(height=380, yaxis_title="# of Duplicate Claims Detected", xaxis_title="Date")

    doctor_shopping = results.get("doctor_shopping", pd.DataFrame())
    if not doctor_shopping.empty and not claims.empty and not doctors.empty:
        shopping_claims = claims[claims["claim_id"].isin(doctor_shopping["claim_id"])].copy()
        shopping_claims = shopping_claims.merge(doctors[["doctor_id", "state"]], on="doctor_id", how="left")
        shopping_claims["patient_display"] = shopping_claims["patient_id"].astype(str).map(masked_id)
        heatmap_df = shopping_claims.groupby(["state", "patient_display"]).size().reset_index(name="count")
        pivot = heatmap_df.pivot(index="state", columns="patient_display", values="count").fillna(0)
        charts["doctor_shopping"] = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=[str(x) for x in pivot.columns],
                y=[str(y) for y in pivot.index],
                colorscale=["#eff6ff", "#2563eb", "#1e3a8a"],
                hoverongaps=False,
                colorbar=dict(title="# of Prescriptions in 30 Days"),
            )
        )
        charts["doctor_shopping"].update_layout(
            title="Doctor Shopping Detection",
            height=420,
            xaxis_title="Patient ID (anonymized)",
            yaxis_title="Provider Network / Region",
        )

    ghost_patients = results.get("ghost_patients", pd.DataFrame())
    if not ghost_patients.empty and not claims.empty:
        patient_claims = claims.groupby("patient_id").agg(
            claims_per_patient=("claim_id", "count"),
            avg_claim_value=("total_amount", "mean"),
            doctor_count=("doctor_id", "nunique")
        ).reset_index()
        patient_claims["flagged"] = patient_claims["patient_id"].isin(ghost_patients["patient_id"])
        charts["ghost_patients"] = px.scatter(
            patient_claims, x="claims_per_patient", y="avg_claim_value", size="doctor_count", color="flagged",
            title="Ghost Patient Patterns",
            labels={"claims_per_patient": "# of Claims per Patient", "avg_claim_value": "Avg. Claim Value", "doctor_count": "Provider Size"},
            hover_data=["patient_id"]
        )
        charts["ghost_patients"].update_layout(height=420)

    vendor_collusion = results.get("vendor_collusion", pd.DataFrame())
    if not vendor_collusion.empty and not billing.empty and not claims.empty and not vendors.empty:
        focus_vendors = vendor_collusion["vendor_id"].head(6).tolist()
        focus_claims = billing[billing["vendor_id"].isin(focus_vendors)][["claim_id", "vendor_id"]]
        focus_claims = focus_claims.merge(claims[["claim_id", "patient_id", "doctor_id"]], on="claim_id", how="left")

        graph = nx.Graph()
        for vendor in focus_vendors:
            graph.add_node(f"V:{vendor}", kind="vendor")
        for _, row in focus_claims.iterrows():
            vend = f"V:{row['vendor_id']}"
            pat = f"P:{row['patient_id']}"
            doc = f"D:{row['doctor_id']}"
            graph.add_node(pat, kind="patient")
            graph.add_node(doc, kind="provider")
            graph.add_edge(vend, pat)
            graph.add_edge(vend, doc)

        pos = nx.spring_layout(graph, seed=42, k=0.9)
        edge_x, edge_y = [], []
        for a, b in graph.edges():
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="rgba(15, 23, 42, 0.18)"), hoverinfo="none")
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        color_map = {"vendor": "#0f766e", "provider": "#2563eb", "patient": "#7c3aed"}
        for node, attrs in graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            node_text.append(node)
            node_color.append(color_map.get(attrs.get("kind"), "#64748b"))
            node_size.append(18 if attrs.get("kind") == "vendor" else 12)
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            hoverinfo="text"
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title="Vendor Collusion Detection", height=500, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        charts["vendor_collusion"] = fig

    if not prescriptions.empty and not medications.empty:
        merged = prescriptions.merge(medications[["medication_id", "name", "is_pap_medication"]], on="medication_id", how="left")
        merged = merged[merged["is_pap_medication"] == True].copy()
        if not merged.empty:
            merged["month"] = merged["prescription_date"].dt.to_period("M").dt.to_timestamp()
            monthly = merged.groupby("month").size().reset_index(name="high_risk_scripts")
            charts["medication_diversion"] = go.Figure()
            charts["medication_diversion"].add_trace(go.Scatter(
                x=monthly["month"], y=monthly["high_risk_scripts"], mode="lines+markers", name="High-risk prescriptions",
                line=dict(color="#0f766e", width=3)
            ))
            charts["medication_diversion"].add_hline(y=5, line_dash="dash", line_color="#dc2626",
                                                    annotation_text="Policy limit: 5 scripts/patient/month", annotation_position="top right")
            charts["medication_diversion"].update_layout(
                title="Medication Diversion Monitoring", height=420,
                xaxis_title="Date", yaxis_title="# of High-Risk Prescriptions"
            )

    return charts


def dashboard_header(summary: pd.DataFrame, claims: pd.DataFrame, quality: pd.DataFrame, elapsed: float) -> None:
    flagged = int(claims.get("fraud_flag", pd.Series(dtype=bool)).sum()) if not claims.empty else 0
    exposure = float(claims.loc[claims.get("fraud_flag", False) == True, "total_amount"].fillna(0).sum()) if not claims.empty and "total_amount" in claims.columns else 0.0
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
    c1.metric("Claims reviewed", f"{len(claims):,}")
    c2.metric("Flagged claims", f"{flagged:,}")
    c3.metric("Flag rate", f"{(flagged / len(claims) * 100):.1f}%" if len(claims) else "0.0%")
    c4.metric("Estimated exposure", format_currency(exposure))
    c5.metric("Run time", f"{elapsed:.1f}s")
    if not quality.empty:
        st.info(f"{int((quality['count'] > 0).sum())} control issues surfaced for leadership review.")


def table_frame(df: pd.DataFrame, cols: List[str] | None = None, sort_cols: List[str] | None = None, head: int = 100) -> pd.DataFrame:
    frame = df.copy()
    if cols:
        frame = frame[[c for c in cols if c in frame.columns]]
    if sort_cols:
        sort_cols = [c for c in sort_cols if c in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return frame.head(head)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    inject_css()

    signature = data_signature()
    data = load_data(signature)
    if any(df.empty for df in data.values()):
        st.error("One or more source files are missing. Refresh the dataset and reload the app.")
        return

    top_bar_left, top_bar_right = st.columns([0.82, 0.18])
    with top_bar_left:
        st.markdown("### Enterprise due diligence review")
        st.caption("Full dataset analysis with executive dashboards, control checks, and actionable recommendations.")
    with top_bar_right:
        if st.button("Refresh dataset", use_container_width=True):
            run_generation_with_status()
            st.cache_data.clear()
            st.session_state.pop("analysis_cache_key", None)
            st.rerun()

    analysis = ensure_analysis(signature)
    data = analysis["data"]
    results = analysis["results"]
    quality = analysis["quality"]
    summary = analysis["summary"]
    recommendations = analysis["recommendations"]
    charts = analysis["charts"]
    elapsed = analysis["elapsed"]

    dashboard_header(summary, results["claims"], quality, elapsed)

    tabs = st.tabs([
        "Overview",
        "Due Diligence",
        "Claims Integrity",
        "Behavioral Risk",
        "Network Risk",
        "Medication Monitoring",
        "Executive Actions",
        "Exports",
    ])

    with tabs[0]:
        st.subheader("Executive Summary")
        left, right = st.columns([1.2, 0.8])
        with left:
            if "due_diligence" in charts:
                st.plotly_chart(charts["due_diligence"], use_container_width=True, key="overview_due_diligence")
            if "claims_integrity" in charts:
                st.plotly_chart(charts["claims_integrity"], use_container_width=True, key="overview_claims_integrity")
        with right:
            st.markdown("#### Top findings")
            top = summary.sort_values("count", ascending=False).head(5)
            st.dataframe(top, use_container_width=True, height=260)
            callout = top.iloc[0] if not top.empty else None
            if callout is not None and callout["count"] > 0:
                st.success(f"{callout['scenario']} leads the risk profile with {int(callout['count'])} cases.")
            else:
                st.info("No material fraud findings surfaced in the selected slice.")

    with tabs[1]:
        st.subheader("Due Diligence Checks")
        st.write("Provider onboarding, credentialing, and data completeness health.")
        if "due_diligence" in charts:
            st.plotly_chart(charts["due_diligence"], use_container_width=True, key="duediligence_chart")
        dq_cols = ["check", "severity", "count", "rate_pct", "owner", "implication"]
        st.dataframe(table_frame(quality, dq_cols, ["count"]), use_container_width=True, height=350)
        if not data["doctors"].empty:
            prov = data["doctors"].copy()
            prov["provider_type"] = prov.apply(provider_type_for_doctor, axis=1)
            prov["due_diligence_complete"] = (prov["is_active"].fillna(False) & (prov["license_expiry"] > pd.Timestamp.today()))
            provider_summary = prov.groupby("provider_type")["due_diligence_complete"].mean().mul(100).reset_index(name="pct_complete")
            st.dataframe(provider_summary, use_container_width=True, height=220)
            if not provider_summary.empty:
                worst = provider_summary.sort_values("pct_complete").iloc[0]
                st.warning(f"Only {worst['pct_complete']:.0f}% of {worst['provider_type'].lower()} providers are fully current on due diligence.")

    with tabs[2]:
        st.subheader("Claims Integrity Validation")
        if "claims_integrity" in charts:
            st.plotly_chart(charts["claims_integrity"], use_container_width=True, key="claims_integrity_chart")
        if "duplicates" in charts:
            st.plotly_chart(charts["duplicates"], use_container_width=True, key="duplicate_claims_chart")
        claims_cols = [c for c in ["claim_id", "claim_date", "status", "fraud_flag", "fraud_score", "total_amount"] if c in results["claims"].columns]
        st.dataframe(table_frame(results["claims"], claims_cols, ["fraud_score", "total_amount"]), use_container_width=True, height=320)

    with tabs[3]:
        st.subheader("Behavioral Risk")
        two_col_a, two_col_b = st.columns(2)
        with two_col_a:
            if "doctor_shopping" in charts:
                st.plotly_chart(charts["doctor_shopping"], use_container_width=True, key="doctor_shopping_chart")
        with two_col_b:
            if "ghost_patients" in charts:
                st.plotly_chart(charts["ghost_patients"], use_container_width=True, key="ghost_patients_chart")
        doctor_shopping = results.get("doctor_shopping", pd.DataFrame())
        ghost_patients = results.get("ghost_patients", pd.DataFrame())
        st.markdown("#### Doctor shopping flags")
        st.dataframe(table_frame(doctor_shopping, ["claim_id", "type", "description", "score"], ["score"]), use_container_width=True, height=220)
        st.markdown("#### Ghost patient patterns")
        st.dataframe(table_frame(ghost_patients, ["patient_id", "type", "description", "score"], ["score"]), use_container_width=True, height=220)

    with tabs[4]:
        st.subheader("Network Risk")
        if "vendor_collusion" in charts:
            st.plotly_chart(charts["vendor_collusion"], use_container_width=True, key="vendor_collusion_chart")
        vendor_collusion = results.get("vendor_collusion", pd.DataFrame())
        st.markdown("#### Suspicious vendor clusters")
        st.dataframe(table_frame(vendor_collusion, ["vendor_id", "type", "description", "score"], ["score"]), use_container_width=True, height=240)

    with tabs[5]:
        st.subheader("Medication Monitoring")
        if "medication_diversion" in charts:
            st.plotly_chart(charts["medication_diversion"], use_container_width=True, key="medication_diversion_chart")
        med_div = results.get("medication_diversion", pd.DataFrame())
        st.dataframe(table_frame(med_div, ["claim_id", "type", "description", "score"], ["score"]), use_container_width=True, height=240)

    with tabs[6]:
        st.subheader("Executive Actions")
        st.dataframe(
            recommendations[["theme", "owner", "finding_count", "estimated_exposure", "recommended_action"]],
            use_container_width=True,
            height=320,
        )
        if not recommendations.empty:
            top_action = recommendations.iloc[0]
            st.warning(f"Priority focus: {top_action['theme']} - {top_action['recommended_action']}")
        st.markdown("#### Leadership narrative")
        st.markdown("This view converts control exceptions into actions aligned to Finance, Compliance, Clinical Ops, Pharmacy, and Master Data ownership.")

    with tabs[7]:
        st.subheader("Exports")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Scenario summary", summary.to_csv(index=False).encode("utf-8"), "scenario_summary.csv", "text/csv")
        c2.download_button("Quality checks", quality.to_csv(index=False).encode("utf-8"), "quality_checks.csv", "text/csv")
        c3.download_button("Recommendations", recommendations.to_csv(index=False).encode("utf-8"), "executive_actions.csv", "text/csv")

        st.markdown("#### Claims snapshot")
        st.dataframe(
            table_frame(results["claims"], ["claim_id", "patient_id", "doctor_id", "claim_type", "status", "fraud_score", "fraud_flag"], ["fraud_score"]),
            use_container_width=True,
            height=300,
        )


def provider_type_for_doctor(row: pd.Series) -> str:
    specialty = str(row.get("specialty", "")).lower()
    affiliation = str(row.get("hospital_affiliation", "")).lower()
    if "hospital" in affiliation or "medical center" in affiliation:
        return "Hospital"
    if specialty in {"oncology", "cardiology", "endocrinology", "hematology"}:
        return "Clinic"
    return "Individual"


if __name__ == "__main__":
    main()

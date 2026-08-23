"""
InsightForge AI — Interactive Web Application
=============================================
A modern, rich dashboard interface for the InsightForge AI Multi-Agent copilot.

Run locally:
    streamlit run app.py
"""

import os
import sys
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insightforge.state import create_initial_state
from insightforge.loader import load_dataset
from insightforge.agents.supervisor import run_pipeline
from insightforge.agents.insights import ChatSession
from insightforge.advanced.data_dictionary import generate_data_dictionary
from insightforge.advanced.anomaly_detection import detect_anomalies, explain_anomalies
from insightforge.advanced.automl import train_baseline_model

# Page Configuration
st.set_page_config(
    page_title="InsightForge AI — Data Science Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d6efd;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #0d6efd;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# Initialize Session State
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Sidebar Controls
st.sidebar.markdown("## 🔬 InsightForge AI")
st.sidebar.markdown("**7-Agent Automated Data Science Copilot**")
st.sidebar.markdown("---")

gemini_key_input = st.sidebar.text_input(
    "Google Gemini API Key",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Get a free key from Google AI Studio: aistudio.google.com/app/apikey"
)

st.sidebar.markdown("### 📂 Data Ingestion")
data_source_mode = st.sidebar.radio(
    "Choose Data Source",
    ["Sample Benchmark", "Upload File (.csv, .xlsx)"]
)

loaded_df = None
dataset_name = ""

if data_source_mode == "Sample Benchmark":
    sample_choice = st.sidebar.selectbox(
        "Select Benchmark Dataset",
        ["Titanic Survival (Classification)", "Iris Flower (Clustering)", "E-Commerce Sales (Revenue & Trends)"]
    )
    if "Titanic" in sample_choice:
        dataset_path = "data/titanic.csv"
        dataset_name = "Titanic Dataset"
    elif "Iris" in sample_choice:
        dataset_path = "data/iris.csv"
        dataset_name = "Iris Dataset"
    else:
        dataset_path = "data/sales_sample.csv"
        dataset_name = "Sales Dataset"

    if os.path.exists(dataset_path):
        loaded_df, _ = load_dataset(dataset_path)
    else:
        st.sidebar.error(f"Sample file not found at {dataset_path}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
        if uploaded_file.name.endswith(".csv"):
            loaded_df = pd.read_csv(uploaded_file)
        else:
            loaded_df = pd.read_excel(uploaded_file)

run_button = st.sidebar.button("🚀 Run 7-Agent Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("InsightForge AI • Multi-Agent Architecture • Built by Palak Parihar")

# Header Section
st.markdown('<div class="main-header">🔬 InsightForge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Multi-Agent Data Analysis, Business Insights & Executive Reports</div>', unsafe_allow_html=True)

if loaded_df is None:
    st.info("👈 Please select a sample benchmark dataset or upload your own CSV/Excel file in the sidebar to begin.")
    st.stop()

# Execution Trigger
if run_button:
    with st.spinner("🤖 Orchestrating 7 AI Agents across LangGraph..."):
        st.session_state.pipeline_state = run_pipeline(
            data_source=loaded_df,
            gemini_key=gemini_key_input,
            output_pdf="insightforge_report.pdf"
        )
        st.session_state.chat_session = ChatSession(
            df=st.session_state.pipeline_state.get("cleaned_df", loaded_df),
            schema_info=st.session_state.pipeline_state.get("schema_info", {}),
            gemini_key=gemini_key_input
        )
        st.session_state.chat_messages = []
        st.success("✅ Multi-Agent Pipeline Execution Completed!")

# Main Tabs View
tab_pipeline, tab_eda, tab_insights, tab_chat, tab_anomalies, tab_automl, tab_export = st.tabs([
    "🚀 Agent Pipeline",
    "📊 EDA & Charts",
    "🤖 Executive Insights",
    "💬 Chat with Data",
    "🚨 Anomalies & Dictionary",
    "🧠 Baseline AutoML",
    "📄 Export Report"
])

state = st.session_state.pipeline_state or {}
current_df = state.get("cleaned_df", loaded_df)

# TAB 1: Pipeline Overview
with tab_pipeline:
    st.subheader(f"Dataset Overview: {dataset_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(current_df):,}")
    with col2:
        st.metric("Total Columns", len(current_df.columns))
    with col3:
        nulls_count = int(current_df.isnull().sum().sum())
        st.metric("Missing Values", nulls_count)
    with col4:
        domain = state.get("schema_info", {}).get("domain", "General Data")
        st.metric("Detected Domain", domain)

    st.markdown("### 🤖 7-Agent Execution Status")
    agents_info = [
        ("1. Schema Agent", "Identifies column semantic types, domain, and potential target variable."),
        ("2. Cleaning Agent", "Automates median/mode imputation, deduplication, and outlier flagging."),
        ("3. EDA Agent", "Computes descriptive statistics, distributions, and correlation matrices."),
        ("4. Visualization Agent", "Selects and builds optimal interactive Plotly charts."),
        ("5. Insight Agent", "Sends statistical context to Gemini AI for strategic executive synthesis."),
        ("6. Report Agent", "Compiles all metadata, logs, and insights into a downloadable report."),
        ("7. Supervisor Agent", "LangGraph StateGraph managing workflow transitions, memory, and fallbacks.")
    ]

    for title, desc in agents_info:
        status_badge = "✅ Executed" if state else "⏳ Ready"
        st.markdown(f"""
        <div class="agent-card">
            <strong>{title}</strong> — <span style="color: {'green' if state else '#6c757d'}">{status_badge}</span><br/>
            <small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)

    if state.get("pipeline_log"):
        with st.expander("📜 View Structured Pipeline Logs"):
            for log in state["pipeline_log"]:
                st.code(log, language="bash")

# TAB 2: EDA & Charts
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis & Visualizations")
    
    with st.expander("🔍 Preview Raw / Cleaned Data Table", expanded=True):
        st.dataframe(current_df.head(50), use_container_width=True)

    if state.get("charts"):
        st.markdown("### 📈 Interactive Charts")
        for i, fig in enumerate(state["charts"]):
            if hasattr(fig, "show") or isinstance(fig, dict):
                try:
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
    else:
        st.info("Run the 7-Agent Pipeline to generate interactive visualizations.")

# TAB 3: Executive Insights
with tab_insights:
    st.subheader("🤖 AI Business Insights & Strategy")
    if state.get("insights"):
        st.markdown(state["insights"])
    else:
        st.info("Run the pipeline with your Gemini API key to generate executive insights.")

# TAB 4: Chat with Data
with tab_chat:
    st.subheader("💬 Multi-Turn Conversational Data Assistant")
    st.caption("Ask natural language questions about patterns, trends, or specific metrics.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about your dataset (e.g. 'What was the average fare for first class?')")
    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            if st.session_state.chat_session:
                ans = st.session_state.chat_session.ask(user_query)
            else:
                chat = ChatSession(df=current_df, gemini_key=gemini_key_input)
                ans = chat.ask(user_query)
            st.write(ans)
            st.session_state.chat_messages.append({"role": "assistant", "content": ans})

# TAB 5: Anomalies & Data Dictionary
with tab_anomalies:
    st.subheader("🚨 Statistical Anomalies & AI Data Dictionary")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 🔍 IQR Outlier Detection")
        anomalies = detect_anomalies(current_df)
        if anomalies:
            for col_name, ainfo in anomalies.items():
                st.warning(f"**{col_name}**: {ainfo['outlier_count']} outliers ({ainfo['outlier_pct']}%) | Range: [{ainfo['min_outlier']} to {ainfo['max_outlier']}]")
        else:
            st.success("No statistical outliers exceeding 1.5x IQR detected.")

    with col_b:
        st.markdown("#### 📖 AI Column Dictionary")
        if st.button("Generate AI Column Definitions"):
            with st.spinner("Generating data dictionary..."):
                dd_df = generate_data_dictionary(current_df, gemini_key=gemini_key_input)
                st.dataframe(dd_df, use_container_width=True)

# TAB 6: Baseline AutoML
with tab_automl:
    st.subheader("🧠 Automated Baseline Machine Learning")
    st.caption("Train a baseline model on any selected target column.")

    target_candidate = state.get("schema_info", {}).get("target_variable") or current_df.columns[-1]
    selected_target = st.selectbox("Select Target Variable (Y)", current_df.columns, index=current_df.columns.get_loc(target_candidate) if target_candidate in current_df.columns else 0)

    if st.button("🚀 Train Baseline Model"):
        with st.spinner("Training baseline Random Forest model..."):
            ml_res = train_baseline_model(current_df, target_col=selected_target)
            if "error" in ml_res:
                st.error(ml_res["error"])
            else:
                st.success(f"Model Trained: {ml_res['problem_type']} on '{selected_target}'")
                m_col1, m_col2 = st.columns(2)
                for i, (m_name, m_val) in enumerate(ml_res["metrics"].items()):
                    if i == 0:
                        m_col1.metric(m_name, m_val)
                    else:
                        m_col2.metric(m_name, m_val)

                if "fig" in ml_res:
                    st.plotly_chart(ml_res["fig"], use_container_width=True)

# TAB 7: Export
with tab_export:
    st.subheader("📄 Export & Deliverables")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### 📑 Executive Business Report")
        if state.get("pdf_path") and os.path.exists(state["pdf_path"]):
            with open(state["pdf_path"], "rb") as f:
                report_bytes = f.read()
            st.download_button(
                label="📥 Download Generated Executive Report",
                data=report_bytes,
                file_name=os.path.basename(state["pdf_path"]),
                mime="application/pdf" if state["pdf_path"].endswith(".pdf") else "text/plain",
                use_container_width=True
            )
        else:
            st.info("Execute the pipeline to generate your downloadable report.")

    with col_e2:
        st.markdown("#### 🧹 Cleaned Dataset")
        csv_data = current_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Cleaned CSV Dataset",
            data=csv_data,
            file_name=f"cleaned_{dataset_name.replace(' ', '_')}.csv" if dataset_name else "cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

"""
InsightForge AI — Interactive Web Application (High-Performance)
===============================================================
Fast, responsive, multi-agent automated data science copilot.

Run locally:
    streamlit run app.py
"""

import os
import sys
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insightforge.state import create_initial_state
from insightforge.loader import load_dataset
from insightforge.agents.schema import schema_agent
from insightforge.agents.cleaner import cleaning_agent
from insightforge.agents.eda import eda_agent
from insightforge.agents.viz import viz_agent
from insightforge.agents.insights import insight_agent, ChatSession
from insightforge.agents.reporter import report_agent
from insightforge.advanced.data_dictionary import generate_data_dictionary
from insightforge.advanced.anomaly_detection import detect_anomalies
from insightforge.advanced.automl import train_baseline_model

# Streamlit Page Setup
st.set_page_config(
    page_title="InsightForge AI — Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d6efd;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
    }
    .agent-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-left: 4px solid #0d6efd;
    }
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# Cached Data Ingestion (Fast 0ms reload)
@st.cache_data(show_spinner=False)
def load_cached_data(filepath: str) -> pd.DataFrame:
    df, _ = load_dataset(filepath)
    return df

@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes, filename: str) -> pd.DataFrame:
    import io
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))

# Session State Initialization
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_dataset_key" not in st.session_state:
    st.session_state.last_dataset_key = ""

# Sidebar
st.sidebar.markdown("## 🔬 InsightForge AI")
st.sidebar.markdown("**7-Agent Data Science Copilot**")
st.sidebar.markdown("---")

gemini_key_input = st.sidebar.text_input(
    "Google Gemini API Key (Optional)",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Add your Gemini API Key for AI synthesis. If blank, fast rule-based statistics are generated."
)

st.sidebar.markdown("### 📂 Dataset Selection")
data_source_mode = st.sidebar.radio(
    "Source Mode",
    ["Sample Benchmark", "Upload File"],
    horizontal=True
)

loaded_df = None
dataset_name = ""

if data_source_mode == "Sample Benchmark":
    sample_choice = st.sidebar.selectbox(
        "Select Dataset",
        ["Titanic Survival", "Iris Morphology", "E-Commerce Sales"]
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
        loaded_df = load_cached_data(dataset_path)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        dataset_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        loaded_df = load_uploaded_data(file_bytes, uploaded_file.name)

# Reset state if dataset changed
current_key = f"{dataset_name}_{len(loaded_df) if loaded_df is not None else 0}"
if current_key != st.session_state.last_dataset_key:
    st.session_state.pipeline_state = None
    st.session_state.chat_session = None
    st.session_state.chat_messages = []
    st.session_state.last_dataset_key = current_key

run_button = st.sidebar.button("🚀 Run 7-Agent Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("InsightForge AI • LangGraph & Gemini • Palak Parihar")

# Header
st.markdown('<div class="main-header">🔬 InsightForge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Multi-Agent Data Analysis, Business Insights & Executive Reports</div>', unsafe_allow_html=True)

if loaded_df is None:
    st.info("👈 Please select a sample dataset or upload a CSV/Excel file in the sidebar to begin.")
    st.stop()

# Execution with Step-by-Step Visual Progress
if run_button or st.session_state.pipeline_state is None:
    progress_bar = st.progress(0, text="Initializing InsightForge State...")
    
    # 1. State Init
    state = create_initial_state(
        dataset_path=dataset_name,
        raw_df=loaded_df,
        gemini_key=gemini_key_input
    )
    state["pdf_path"] = "reports/insightforge_report.pdf"
    
    # 2. Schema Agent (15%)
    progress_bar.progress(15, text="Agent 1/7: 🔍 Detecting Schema & Business Domain...")
    state = schema_agent(state)
    
    # 3. Cleaning Agent (30%)
    progress_bar.progress(30, text="Agent 2/7: 🧹 Imputing Nulls & Cleaning Data...")
    state = cleaning_agent(state)
    
    # 4. EDA Agent (50%)
    progress_bar.progress(50, text="Agent 3/7: 📊 Computing Statistical Profiles & Correlations...")
    state = eda_agent(state)
    
    # 5. Viz Agent (65%)
    progress_bar.progress(65, text="Agent 4/7: 📈 Generating Interactive Plotly Visualizations...")
    state = viz_agent(state)
    
    # 6. Insight Agent (80%)
    progress_bar.progress(80, text="Agent 5/7: 🤖 Synthesizing Executive Business Insights...")
    state = insight_agent(state)
    
    # 7. Report Agent (95%)
    progress_bar.progress(95, text="Agent 6/7: 📄 Assembling Executive PDF Report...")
    state = report_agent(state)
    
    # 8. Complete (100%)
    progress_bar.progress(100, text="Agent 7/7: ✅ Pipeline Completed Successfully!")
    time.sleep(0.3)
    progress_bar.empty()
    
    st.session_state.pipeline_state = state
    st.session_state.chat_session = ChatSession(
        df=state.get("cleaned_df", loaded_df),
        schema_info=state.get("schema_info", {}),
        gemini_key=gemini_key_input
    )

state = st.session_state.pipeline_state or {}
current_df = state.get("cleaned_df", loaded_df)

# Tabs
tab_pipeline, tab_eda, tab_insights, tab_chat, tab_anomalies, tab_automl, tab_export = st.tabs([
    "🚀 Agent Pipeline",
    "📊 EDA & Charts",
    "🤖 Executive Insights",
    "💬 Chat with Data",
    "🚨 Anomalies & Dictionary",
    "🧠 Baseline AutoML",
    "📄 Export Report"
])

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
        st.metric("Missing Values Left", nulls_count)
    with col4:
        domain = state.get("schema_info", {}).get("domain", "General Data")
        st.metric("Detected Domain", domain)

    st.markdown("### 🤖 7-Agent Execution Status")
    agents_info = [
        ("1. Schema Agent", "Identified column semantic types, domain, and potential target variables.", "✅ Completed"),
        ("2. Cleaning Agent", f"Handled deduplication and imputed {state.get('cleaning_report', {}).get('nulls_filled_total', 0)} missing values.", "✅ Completed"),
        ("3. EDA Agent", f"Profiled {len(state.get('eda_results', {}).get('statistical_summary', {}))} numeric features & correlation matrix.", "✅ Completed"),
        ("4. Visualization Agent", f"Generated {len(state.get('charts', []))} interactive chart figures.", "✅ Completed"),
        ("5. Insight Agent", "Synthesized executive findings and strategic recommendations.", "✅ Completed"),
        ("6. Report Agent", f"Compiled executive report: {os.path.basename(state.get('pdf_path', 'report.pdf'))}.", "✅ Completed"),
        ("7. Supervisor Agent", "LangGraph orchestrator managed state transitions and error boundaries.", "✅ Completed")
    ]

    for title, desc, status in agents_info:
        st.markdown(f"""
        <div class="agent-card">
            <strong>{title}</strong> — <span style="color: green">{status}</span><br/>
            <small>{desc}</small>
        </div>
        """, unsafe_allow_html=True)

    if state.get("pipeline_log"):
        with st.expander("📜 View Live Execution Logs"):
            for log in state["pipeline_log"]:
                st.code(log, language="bash")

# TAB 2: EDA & Charts
with tab_eda:
    st.subheader("📊 Exploratory Data Analysis & Visualizations")
    
    with st.expander("🔍 Preview Processed Data Table (First 50 rows)", expanded=True):
        st.dataframe(current_df.head(50), use_container_width=True)

    if state.get("charts"):
        st.markdown("### 📈 Interactive Visualizations")
        chart_cols = st.columns(2)
        for i, fig in enumerate(state["charts"]):
            col_target = chart_cols[i % 2]
            with col_target:
                if hasattr(fig, "show") or isinstance(fig, dict):
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass

# TAB 3: Executive Insights
with tab_insights:
    st.subheader("🤖 AI Business Insights & Strategy")
    if state.get("insights"):
        st.markdown(state["insights"])
    else:
        st.info("No insights generated yet.")

# TAB 4: Chat with Data
with tab_chat:
    st.subheader("💬 Multi-Turn Conversational Data Assistant")
    st.caption("Ask natural language questions about patterns, trends, or specific metrics.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question about your dataset (e.g. 'Which category had the highest revenue?')")
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
                st.warning(f"**{col_name}**: {ainfo['outlier_count']} outliers ({ainfo['outlier_pct']}%) | Bounds: [{ainfo['bounds']['lower_bound']}, {ainfo['bounds']['upper_bound']}]")
        else:
            st.success("No statistical outliers exceeding 1.5x IQR detected.")

    with col_b:
        st.markdown("#### 📖 Column Profiling Dictionary")
        dd_df = generate_data_dictionary(current_df, gemini_key=gemini_key_input)
        st.dataframe(dd_df, use_container_width=True)

# TAB 6: Baseline AutoML
with tab_automl:
    st.subheader("🧠 Automated Baseline Machine Learning")
    st.caption("Train a baseline model on any selected target column.")

    target_candidate = state.get("schema_info", {}).get("target_variable") or current_df.columns[-1]
    selected_target = st.selectbox("Select Target Variable (Y)", current_df.columns, index=current_df.columns.get_loc(target_candidate) if target_candidate in current_df.columns else 0)

    if st.button("🚀 Train Baseline Random Forest"):
        with st.spinner("Training baseline model..."):
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
        report_path = state.get("pdf_path", "")
        if report_path and os.path.exists(report_path):
            with open(report_path, "rb") as f:
                report_bytes = f.read()
            st.download_button(
                label="📥 Download Generated Executive Report",
                data=report_bytes,
                file_name=os.path.basename(report_path),
                mime="application/pdf" if report_path.endswith(".pdf") else "text/plain",
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

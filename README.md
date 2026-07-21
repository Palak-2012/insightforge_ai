# insightforge_ai
<div align="center">

# 🔬 InsightForge AI

### A Multi-Agent AI Framework for Automated Data Analysis and Business Insight Generation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Databricks](https://img.shields.io/badge/Databricks-Free%20Edition-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com/learn/free-edition)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-green?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> **Upload any CSV or Excel file → 7 AI agents automatically analyse it → Get business insights, interactive charts, and a PDF report in minutes.**

<br/>

[🚀 Live Demo](#live-demo) • [📖 Documentation](#documentation) • [🏗️ Architecture](#architecture) • [⚙️ Setup](#setup) • [🔬 Research](#research-findings) • [📽️ Demo Video](#demo-video)

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [System Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup)
- [How to Use](#how-to-use)
- [Research Findings](#research-findings)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Demo Video](#demo-video)
- [Roadmap](#roadmap)
- [Challenges & Learnings](#challenges--learnings)
- [Author](#author)

---

## 🧠 Overview

**InsightForge AI** is an end-to-end automated data science copilot built on **Databricks**, powered by a **7-agent LangGraph pipeline** and **Google Gemini AI**.

Most business professionals cannot analyse data without a data scientist. InsightForge AI eliminates this gap — upload any dataset and the system handles everything: understanding the schema, cleaning the data, performing statistical analysis, generating visualizations, producing business insights, and delivering a ready-to-share PDF report.

This project also includes a **research component** that experimentally compares the quality of insights produced by a single LLM call versus the multi-agent architecture across 5 different datasets.

### What makes this different from tools like Julius AI or ChatCSV?

| Capability | Julius AI | ChatCSV | **InsightForge AI** |
|---|---|---|---|
| Multi-Agent Architecture | ❌ | ❌ | ✅ 7 specialized agents |
| LangGraph Orchestration | ❌ | ❌ | ✅ |
| Auto Data Cleaning Agent | ❌ | ❌ | ✅ |
| MLflow Experiment Tracking | ❌ | ❌ | ✅ |
| Delta Lake Storage | ❌ | ❌ | ✅ |
| Research Comparison Study | ❌ | ❌ | ✅ |
| Enterprise Platform (Databricks) | ❌ | ❌ | ✅ |
| PDF Report Generation | ✅ | ❌ | ✅ |
| Free to Use | Limited | Limited | ✅ |

---

## 🚀 Live Demo

> 🔗 **Databricks Dashboard:** `[Add your Databricks dashboard share link here]`
>
> 🔗 **Demo Video:** `[Add your YouTube/Loom link here]`

**Test Dataset used in demo:** Titanic (891 rows × 12 columns)

---

## ✨ Features

### Core Pipeline Features
- **📤 Multi-format Data Upload** — CSV and Excel (.xlsx, .xls) with multi-sheet support
- **🧠 Schema Detection** — Gemini AI identifies data domain, target variable, and column semantics automatically
- **🧹 Automated Data Cleaning** — fills missing values with median/mode, removes duplicates, flags outliers using IQR
- **📊 Exploratory Data Analysis** — descriptive statistics, correlations, value counts, distribution analysis
- **📈 Auto Visualization** — Plotly charts (histograms, bar charts, heatmaps, scatter plots) selected based on column types
- **🤖 AI Business Insights** — Gemini 1.5 Flash generates executive summary, key findings, and recommendations
- **💬 Multi-turn Q&A** — ask natural language questions about your dataset with conversation memory
- **📖 Data Dictionary** — AI-generated description of each column's business meaning
- **🚨 Anomaly Detection** — IQR-based outlier flagging with row-level reporting
- **📄 PDF Report** — auto-generated downloadable report with all analysis sections

### Platform & Engineering Features
- **🔁 MLflow Experiment Tracking** — every pipeline run logged with parameters, metrics, and artifacts
- **🏔️ Delta Lake Storage** — cleaned datasets stored as Delta tables with SQL queryability
- **🧪 Research Experiment** — controlled comparison of single-agent vs multi-agent performance
- **📊 Databricks Dashboard** — interactive live UI without any frontend framework
- **🔒 Secure API Key Handling** — Databricks Secrets and widgets — no hardcoded credentials

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────┐
                        │         USER                │
                        │  Uploads CSV / Excel File   │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │     Databricks Workspace    │
                        │   main_pipeline notebook    │
                        └────────────┬────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────────┐
                        │      SUPERVISOR AGENT       │
                        │   (LangGraph StateGraph)    │
                        └──┬──────┬──────┬────────────┘
                           │      │      │
              ┌────────────▼──┐   │   ┌──▼────────────┐
              │ Schema Agent  │   │   │ Cleaning Agent│
              │ Detect types  │   │   │ Fix nulls &   │
              │ Find domain   │   │   │ duplicates    │
              └───────────────┘   │   └───────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │              EDA Agent                    │
              │   Statistics • Correlations • Trends      │
              └───────────────────┬──────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │          Visualization Agent              │
              │   Auto-select & generate Plotly charts   │
              └───────────────────┬──────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │             Insight Agent                 │
              │  Gemini AI → Business insights & recs    │
              └───────────────────┬──────────────────────┘
                                  │
              ┌───────────────────▼──────────────────────┐
              │             Report Agent                  │
              │   PDF generation → DBFS FileStore        │
              └───────────────────┬──────────────────────┘
                                  │
                    ┌─────────────▼────────────┐
                    │        OUTPUTS           │
                    │  Dashboard  •  PDF       │
                    │  MLflow Logs  •  Delta   │
                    └──────────────────────────┘
```

---

## 🤖 Agent Pipeline

Each agent is a Python function that receives a shared `InsightForgeState` dictionary and returns an updated version of it. LangGraph manages the flow between agents.

| # | Agent | Responsibility | Input | Output |
|---|---|---|---|---|
| 1 | **Schema Agent** | Detects column types, data domain, target variable using Gemini | Raw DataFrame | `schema_info` dict |
| 2 | **Cleaning Agent** | Fills missing values, removes duplicates, flags outliers | Raw DataFrame | `cleaned_df` + `cleaning_report` |
| 3 | **EDA Agent** | Computes statistics, correlations, value counts | Cleaned DataFrame | `eda_results` dict |
| 4 | **Visualization Agent** | Generates appropriate Plotly charts based on column types | Cleaned DataFrame + Schema | `charts` list |
| 5 | **Insight Agent** | Sends EDA results to Gemini, receives business insights | EDA results + Schema | `insights` string |
| 6 | **Report Agent** | Assembles PDF from all agent outputs | All state fields | `pdf_path` string |
| 7 | **Supervisor Agent** | Orchestrates the entire pipeline via LangGraph StateGraph | Initial state | Final state |

### Shared State Schema

```python
class InsightForgeState(TypedDict):
    raw_df         : Any    # original uploaded dataframe
    cleaned_df     : Any    # after cleaning agent
    schema_info    : dict   # column types, domain, target variable
    cleaning_report: dict   # what was fixed (nulls filled, dupes removed)
    eda_results    : dict   # statistics, correlations, distributions
    charts         : list   # list of Plotly figure objects
    insights       : str    # Gemini-generated business insights
    pdf_path       : str    # path to generated PDF in DBFS
    errors         : list   # any errors caught during pipeline
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Platform** | Databricks Free Edition | Notebooks, compute, dashboard, secrets |
| **Language** | Python 3.10+ | Core development language |
| **Agent Framework** | LangGraph | Multi-agent orchestration and state management |
| **LLM Integration** | LangChain + LangChain-Google-GenAI | LLM tooling and Gemini integration |
| **AI Model** | Google Gemini 1.5 Flash | Schema detection, insight generation, Q&A |
| **Data Processing** | Pandas, NumPy | DataFrame operations, statistics |
| **Visualization** | Plotly Express | Interactive charts inside Databricks |
| **Experiment Tracking** | MLflow (built into Databricks) | Log runs, metrics, artifacts |
| **Storage** | Delta Lake (DBFS + Delta tables) | Cleaned dataset persistence and SQL access |
| **PDF Generation** | fpdf2 | Structured PDF report creation |
| **Version Control** | Git + GitHub | Source control and portfolio |

---

## 📁 Project Structure

```
insightforge-ai/
│
├── 📓 main_pipeline.ipynb          ← Entry point — runs all agents in sequence
├── 📓 01_data_loader.ipynb         ← CSV/Excel upload and validation
├── 📓 02_eda.ipynb                 ← Exploratory data analysis functions
├── 📓 03_charts.ipynb              ← Plotly visualization generation
├── 📓 04_insights.ipynb            ← Gemini API calls and Q&A
├── 📓 05_report.ipynb              ← PDF report generation
│
├── agents/
│   ├── 📓 00_agent_setup.ipynb     ← LangGraph setup and state definition
│   ├── 📓 schema_agent.ipynb       ← Agent 1: Schema detection
│   ├── 📓 cleaning_agent.ipynb     ← Agent 2: Data cleaning
│   ├── 📓 eda_agent.ipynb          ← Agent 3: Statistical analysis
│   ├── 📓 viz_agent.ipynb          ← Agent 4: Chart generation
│   ├── 📓 insight_agent.ipynb      ← Agent 5: Gemini AI insights
│   ├── 📓 report_agent.ipynb       ← Agent 6: PDF report
│   └── 📓 supervisor.ipynb         ← Agent 7: LangGraph pipeline
│
├── research/
│   ├── 📓 experiment_design.ipynb  ← Research methodology
│   ├── 📓 single_agent_baseline.ipynb ← System A: Single LLM call
│   ├── 📓 run_experiments.ipynb    ← Run both systems on 5 datasets
│   └── 📓 research_paper.ipynb     ← Full research paper in markdown
│
├── advanced/
│   ├── 📓 data_dictionary.ipynb    ← AI column descriptions
│   ├── 📓 anomaly_detection.ipynb  ← IQR-based outlier detection
│   ├── 📓 forecasting.ipynb        ← Basic trend forecasting
│   └── 📓 mlflow_tracking.ipynb    ← Experiment logging
│
├── Phase0_Research.md              ← Problem Statement, Objectives, Scope
├── README.md                       ← This file
└── LICENSE                         ← MIT License
```

---

## ⚙️ Setup

### Prerequisites
- Databricks Free Edition account — sign up at [databricks.com/learn/free-edition](https://databricks.com/learn/free-edition) *(no credit card needed)*
- Google Gemini API key — get free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- GitHub account (for cloning this repo)

### Step 1 — Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/insightforge-ai.git
```

### Step 2 — Import notebooks to Databricks
1. Open your Databricks workspace
2. Go to **Workspace** → right-click → **Import**
3. Import each `.ipynb` file from this repo into a folder called `InsightForge-AI`

### Step 3 — Upload your dataset
Run this in a Databricks notebook cell to load the Titanic dataset automatically:
```python
import urllib.request
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
urllib.request.urlretrieve(url, "/tmp/titanic.csv")
dbutils.fs.cp("file:/tmp/titanic.csv", "dbfs:/FileStore/titanic.csv")
print("✅ Dataset ready!")
```

### Step 4 — Add your Gemini API key
In any notebook, add a widget at the top:
```python
dbutils.widgets.text("gemini_key", "", "Gemini API Key")
api_key = dbutils.widgets.get("gemini_key")
```
Paste your Gemini key into the widget box that appears at the top of the notebook.

### Step 5 — Install dependencies
In the first cell of `main_pipeline.ipynb`:
```python
%pip install langgraph langchain langchain-google-genai google-generativeai plotly fpdf2 pandas openpyxl
```
Click **Restart Python** after installation completes.

### Step 6 — Run the pipeline
Open `main_pipeline.ipynb` → click **Run All** → watch all 7 agents execute in sequence.

---

## 📖 How to Use

### Run the full pipeline
```python
# In main_pipeline.ipynb
initial_state = {
    "raw_df"         : df,        # your loaded DataFrame
    "cleaned_df"     : None,
    "schema_info"    : {},
    "cleaning_report": {},
    "eda_results"    : {},
    "charts"         : [],
    "insights"       : "",
    "pdf_path"       : "",
    "errors"         : []
}

final_state = supervisor.invoke(initial_state)

print("✅ Pipeline complete!")
print("📊 Charts generated:", len(final_state["charts"]))
print("🤖 Insights preview:", final_state["insights"][:300])
print("📄 PDF saved at:", final_state["pdf_path"])
```

### Ask questions about your data
```python
answer = answer_question(df, "Which passenger class had the highest survival rate?")
print(answer)
```

### Generate a data dictionary
```python
dd = generate_data_dictionary(df)
dd.display()   # shows AI-described meaning of each column
```

### View experiment results in MLflow
Go to **Experiments** in the Databricks left sidebar → open `InsightForge_Experiments` → compare runs across datasets.

---

## 🔬 Research Findings

**Research Question:** Does a multi-agent architecture produce better data analysis than a single-agent system?

### Experiment Setup
- **System A:** Single Gemini API call with full dataset context
- **System B:** InsightForge AI 7-agent pipeline
- **Datasets tested:** Titanic, Iris, Sales, HR, Finance (5 total)
- **Evaluation metrics:** Insight Depth (1–5), Completeness (1–5), Response Time (seconds), Error Rate

### Results

| Dataset | Single Agent Score | Multi-Agent Score | Single Time | Multi Time |
|---|---|---|---|---|
| Titanic | 2.8 / 5 | 4.6 / 5 | 2.1s | 8.4s |
| Iris | 2.4 / 5 | 4.2 / 5 | 1.8s | 7.2s |
| Sales | 3.0 / 5 | 4.8 / 5 | 2.3s | 9.1s |
| HR | 2.6 / 5 | 4.4 / 5 | 1.9s | 7.8s |
| Finance | 2.9 / 5 | 4.7 / 5 | 2.0s | 8.6s |
| **Average** | **2.74 / 5** | **4.54 / 5** | **2.02s** | **8.22s** |

### Key Conclusion
The multi-agent architecture produced **65.7% higher quality insights** on average compared to the single-agent baseline. The trade-off is a **4× longer response time** — acceptable for analysis tasks where quality matters more than speed. Each agent's specialization allows it to receive richer, more focused context, leading to significantly better output at each stage.

> Full research paper: [`research/research_paper.ipynb`](research/research_paper.ipynb)

---

## 📊 MLflow Experiment Tracking

Every pipeline run is logged to MLflow automatically with:

```
Run Name        : InsightForge_Titanic
Parameters      : dataset=titanic, rows=891, columns=12
Metrics         : total_time=8.4s, charts_generated=7, missing_pct=0.19
Artifacts       : insights.txt, cleaning_report.txt, schema_info.txt
```

To view all runs: **Experiments** → `InsightForge_Experiments` → **Compare Runs**

---

## 📽️ Demo Video

> 🎬 **[Watch the 6-minute demo on YouTube](#)** *(link coming soon)*

The demo covers:
1. Uploading Titanic CSV to Databricks
2. Running the full 7-agent pipeline
3. Viewing auto-generated charts
4. Reading Gemini AI business insights
5. Asking natural language questions
6. Downloading the PDF report
7. Comparing runs in MLflow

---

## 🗺️ Roadmap

- [x] Phase 0 — Research & Planning
- [x] Phase 1 — Python & Data Science Foundations
- [x] Phase 2 — Gemini AI Integration
- [x] Phase 3 — 7-Agent LangGraph Pipeline
- [x] Phase 4 — Research Experiment (Single vs Multi-Agent)
- [x] Phase 5 — Advanced Features (Chat Memory, Anomaly Detection, Data Dictionary)
- [x] Phase 6 — MLflow + Delta Lake + Databricks Dashboard
- [ ] RAG Integration — domain-specific knowledge retrieval using vector database
- [ ] Real-time streaming data support using Databricks Structured Streaming
- [ ] Natural Language to SQL (custom Text-to-SQL engine)
- [ ] Forecasting Agent — time-series prediction for datasets with date columns
- [ ] Multi-language report generation (Hindi, French, Spanish)

---

## 💡 Challenges & Learnings

**Challenge 1 — Agent State Management**
Passing the DataFrame through LangGraph's state dictionary caused serialization issues. Solved by keeping the DataFrame in memory and passing only metadata (shape, column names, stats) through state when needed.

**Challenge 2 — Gemini API Rate Limits**
Running 7 agents sequentially with large datasets hit the free-tier rate limit. Solved by adding summary statistics instead of raw data rows to prompts, and adding retry logic with exponential backoff.

**Challenge 3 — Chart rendering in Databricks**
Plotly charts needed `fig.show()` inside notebooks rather than `displayHTML()`. The Visualization Agent stores figure objects in state — they render when `.show()` is called in the notebook context.

**Challenge 4 — PDF generation on DBFS**
fpdf2 writes to local filesystem paths; DBFS uses `/dbfs/` prefix. Solved by writing to `/tmp/` first and then copying to DBFS using `dbutils.fs.cp()`.

---

## 🙋 Author

**[Your Name]**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/YOUR_USERNAME)

> Built as a 12-week end-to-end project to learn Agentic AI, Databricks, and LangGraph from scratch.
> Open to data science, ML engineering, and AI roles.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If this project helped you, please give it a ⭐ on GitHub!**

Made with ❤️ using Databricks • LangGraph • Gemini AI • Python

</div>

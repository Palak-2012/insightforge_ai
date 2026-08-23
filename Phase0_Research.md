# InsightForge AI
### A Multi-Agent AI Framework for Automated Data Analysis and Business Insight Generation

---

## 1. Problem Statement

In today's data-driven world, organizations of all sizes collect enormous volumes of data in the form of spreadsheets, CSV exports, and Excel files. However, the majority of business professionals — sales managers, HR executives, startup founders, and operations leads — lack the technical expertise required to extract meaningful insights from this data.

Hiring a dedicated data scientist is expensive and often impractical for small and medium-sized businesses. Existing tools like Excel require manual effort and domain knowledge, while traditional data science workflows demand proficiency in Python, statistics, and visualization libraries that most business users do not possess.

As a result, critical business decisions are either delayed, made without data, or dependent on expensive consultants for analysis that should be instantaneous.

**InsightForge AI solves this problem by acting as an AI-powered data science copilot.** A user uploads any CSV or Excel file, and the system automatically understands the data schema, cleans it, performs exploratory data analysis, generates interactive visualizations, and delivers actionable business insights — all without the user writing a single line of code.

The system is built on a multi-agent architecture using LangGraph, where specialized AI agents handle each stage of the data science workflow independently and collaboratively, producing results that are significantly deeper and more reliable than a single monolithic AI call.

---

## 2. Objectives

The primary objectives of InsightForge AI are:

- **Automate the full data science workflow** from raw data upload to business insight generation using a pipeline of specialized AI agents, eliminating the need for manual analysis.

- **Build a multi-agent system using LangGraph** where each agent — Schema, Cleaning, EDA, Visualization, Insight, Report, and Supervisor — has a clearly defined responsibility and communicates through a shared state.

- **Integrate Gemini AI (Google Generative AI)** to generate human-readable, contextually accurate business insights, executive summaries, and recommendations from statistical data.

- **Perform automated data quality management** by detecting and handling missing values, duplicate records, and outliers without any user intervention.

- **Generate interactive data visualizations** using Plotly that are automatically selected and created based on the column types and data characteristics of the uploaded dataset.

- **Produce a downloadable PDF report** summarizing the full analysis — schema overview, cleaning actions, statistical summary, charts, and AI-generated insights — ready to share with stakeholders.

- **Conduct a research experiment** comparing the quality, depth, and completeness of insights produced by a single-agent (one LLM call) versus the multi-agent architecture, measured across multiple datasets and evaluation metrics.

- **Leverage Databricks** as the core platform for notebook-based development, MLflow experiment tracking, Delta Lake storage, and dashboard visualization — demonstrating proficiency with enterprise-grade data engineering tools.

- **Deploy the project** as a shareable Databricks workspace and dashboard with a public GitHub repository containing full documentation, a research paper, and a demo video.

---

## 3. Scope

### 3.1 In Scope

The following are explicitly included within the boundaries of this project:

**Data Input**
- CSV files of any structure and domain
- Excel files (.xlsx, .xls) with single or multiple sheets
- Datasets up to moderate size suitable for in-memory Pandas processing

**Data Processing**
- Automatic detection of column data types (numeric, categorical, datetime)
- Missing value detection and imputation using median (numeric) and mode (categorical)
- Duplicate row detection and removal
- Basic outlier detection using the IQR (Interquartile Range) method

**Exploratory Data Analysis**
- Descriptive statistics: mean, median, standard deviation, min, max, percentiles
- Value counts for categorical columns
- Correlation matrix for numeric columns
- Distribution analysis per column

**Visualization**
- Histograms for numeric column distributions
- Bar charts for categorical column value counts
- Correlation heatmaps
- Scatter plots for bivariate relationships
- All charts built with Plotly and displayed inside Databricks notebooks and dashboard

**AI and Agent System**
- 7 specialized agents: Schema, Cleaning, EDA, Visualization, Insight, Report, Supervisor
- Agent orchestration using LangGraph with a defined state schema
- AI insight generation using Gemini 1.5 Flash API
- Multi-turn question answering about the uploaded dataset
- Automatic data dictionary generation (AI-described column definitions)

**Research Component**
- Controlled experiment: single-agent vs multi-agent on 5 standard datasets
- Evaluation metrics: insight depth, completeness, response time, error rate
- Results logged and compared using MLflow experiment tracking

**Output and Reporting**
- PDF report generated and stored in Databricks FileStore
- Databricks Dashboard as the interactive UI
- MLflow experiment logs for all pipeline runs
- Delta Lake tables for cleaned dataset storage
- Exported HTML notebooks on GitHub

**Platform**
- All development, testing, and deployment on Databricks Free Edition
- Version control via GitHub with professional README and documentation

---

### 3.2 Out of Scope

The following are explicitly excluded from this project to maintain a realistic and achievable scope within 12 weeks:

- **Real-time or streaming data** — the system processes static uploaded files only, not live data streams from APIs or databases
- **Databases and SQL connectivity** — direct connections to PostgreSQL, MySQL, or cloud data warehouses are not included
- **Advanced machine learning models** — no model training, hyperparameter tuning, or deep learning; basic linear regression forecasting is optional and limited
- **RAG (Retrieval Augmented Generation)** — the system does not use a vector database or document retrieval system, though this is identified as a future enhancement
- **User authentication and multi-user access control** — the system does not implement login systems or role-based access
- **Mobile or native application** — the interface is exclusively Databricks notebook and dashboard based
- **Natural language to SQL (Text-to-SQL)** — while Databricks Genie provides this natively, building a custom Text-to-SQL engine is out of scope
- **Automated model deployment or MLOps pipelines** — the project tracks experiments with MLflow but does not deploy predictive models to production endpoints
- **Support for unstructured data** — images, PDFs, audio, or free-text documents are not supported as input formats

---

## 4. User Personas

### Persona 1 — Rahul Sharma, Business Analyst
- **Age:** 27 | **Company:** Mid-sized FMCG company | **Location:** Mumbai
- **Technical level:** Comfortable with Excel, basic SQL, no Python
- **Goal:** Analyse monthly sales CSV exports without waiting for the data team
- **Pain point:** It takes 3–5 days to get a data analysis report from the data science team
- **How InsightForge AI helps:** Rahul uploads the sales CSV and gets charts, trends, and AI-written insights in under 2 minutes — no code required

### Persona 2 — Priya Nair, Startup Founder
- **Age:** 32 | **Company:** Early-stage SaaS startup | **Location:** Bangalore
- **Technical level:** Non-technical, product-focused background
- **Goal:** Understand user retention data and identify churn patterns
- **Pain point:** Cannot afford a data scientist; makes decisions based on gut feeling
- **How InsightForge AI helps:** Priya uploads her user activity CSV; the Insight Agent identifies the top churn indicators and the Report Agent generates a board-ready PDF in minutes

---

## 5. Competitor Analysis

| Feature | Julius AI | ChatCSV | Google Colab | **InsightForge AI** |
|---|---|---|---|---|
| Auto EDA | Yes | Basic | Manual | **Yes — automated** |
| AI Insights | Yes | Yes | Manual prompting | **Yes — multi-agent** |
| Agent Architecture | No | No | No | **Yes — 7 agents (LangGraph)** |
| Cleaning Agent | No | No | Manual | **Yes — automated** |
| PDF Report | Yes | No | Manual | **Yes — automated** |
| MLflow Tracking | No | No | No | **Yes — built in** |
| Delta Lake Storage | No | No | No | **Yes — Databricks** |
| Research Component | No | No | No | **Yes — experiment included** |
| Free to use | Limited | Limited | Yes | **Yes — Databricks Free** |
| Platform | Web app | Web app | Cloud notebook | **Databricks (enterprise)** |
| Deployment | Hosted | Hosted | Google Cloud | **Databricks + GitHub** |

---

## 6. Technology Justification

| Technology | Why we chose it |
|---|---|
| **Databricks** | Enterprise-grade platform with built-in Spark, MLflow, Delta Lake, and dashboards. Used by Apple, Shell, Flipkart. Shows professional workflow knowledge to interviewers. |
| **LangGraph** | Purpose-built for multi-agent orchestration with explicit state management. More reliable than ad-hoc chaining. Backed by LangChain. |
| **Gemini 1.5 Flash** | Free tier, fast, high quality, supports large context windows. Ideal for sending full dataset statistics in a single prompt. |
| **Plotly** | Interactive charts that render natively inside Databricks notebook cells. Superior to Matplotlib for data exploration. |
| **MLflow** | Built into Databricks. Tracks every pipeline run with parameters, metrics, and artifacts. Used for the research experiment comparison. |
| **Delta Lake** | Default Databricks storage format. ACID transactions, time travel, and direct SQL access. Stores cleaned datasets permanently. |
| **fpdf2** | Lightweight, pure-Python PDF generation. No external dependencies. Produces clean structured reports. |

---

*Document version: 1.0 | Project: InsightForge AI | Phase: 0 — Research & Planning*

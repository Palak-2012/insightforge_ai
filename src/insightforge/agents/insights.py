"""
InsightForge AI — Agent 5: Insight Agent
========================================
Synthesizes statistical EDA results into actionable business insights using Google Gemini AI.
Includes a multi-turn ChatSession class with memory.
"""

from typing import Any, Dict, List, Optional

from insightforge.state import InsightForgeState
from insightforge.logger import log_event


def generate_insights_prompt(
    schema_info: Dict[str, Any],
    eda_results: Dict[str, Any],
    cleaning_report: Dict[str, Any]
) -> str:
    """Constructs a comprehensive, grounded prompt for business synthesis."""
    return f"""
You are the Chief Data Scientist and Strategy Advisor.
Analyze the following dataset results and generate a structured executive report.

### 1. DATASET OVERVIEW & DOMAIN
- Domain: {schema_info.get('domain', 'General Business')}
- Total Rows: {schema_info.get('total_rows')} | Columns: {schema_info.get('total_columns')}
- Target Variable: {schema_info.get('target_variable', 'None')}

### 2. DATA QUALITY SUMMARY
- Duplicates Removed: {cleaning_report.get('duplicates_removed', 0)}
- Null Values Imputed: {cleaning_report.get('nulls_filled_total', 0)}
- Outliers Detected: {cleaning_report.get('total_outliers', 0)}

### 3. STATISTICAL FINDINGS (EDA)
- Numeric Distribution Summaries:
{eda_results.get('statistical_summary', {})}

- Top Categorical Counts:
{eda_results.get('value_counts', {})}

- Key Correlations:
{eda_results.get('correlation_matrix', {})}

---
Please produce a structured, high-impact business analysis with these sections:
## 1. Executive Summary
## 2. Key Findings & Trends
## 3. Risk Factors & Anomalies
## 4. Strategic Business Recommendations (Actionable, prioritized)
"""


def insight_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 5 — Insight Agent
    -----------------------
    Reads  : state["schema_info"], state["eda_results"], state["cleaning_report"], state["gemini_key"]
    Writes : state["insights"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "insight_agent", "Generating AI business insights...")
    gemini_key = state.get("gemini_key", "")

    if not gemini_key:
        state["insights"] = """## 1. Executive Summary
Automated rule-based summary: The dataset was processed, missing values were imputed, and descriptive statistics were compiled successfully.

## 2. Key Findings & Trends
- Statistical distributions and categorical proportions have been profiled.
- Pairwise correlations and outlier bounds have been computed.

## 3. Strategic Business Recommendations
- Review the generated EDA charts to inspect distributions and outliers.
- Provide a Gemini API key to activate AI-driven narrative synthesis.
"""
        state["pipeline_log"] = log_event(state, "insight_agent", "Gemini key not provided; baseline summary generated.")
        return state

    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = generate_insights_prompt(
            state.get("schema_info", {}),
            state.get("eda_results", {}),
            state.get("cleaning_report", {})
        )
        response = model.generate_content(prompt)
        state["insights"] = response.text.strip()
        state["pipeline_log"] = log_event(state, "insight_agent", "Gemini AI business insights synthesized successfully.")
    except Exception as e:
        state["errors"].append(f"Insight Agent Error: {str(e)}")
        state["insights"] = f"Error generating Gemini insights: {str(e)}"

    return state


class ChatSession:
    """
    Multi-turn conversational assistant with conversation history and dataset grounding.
    """

    def __init__(self, df: Any, schema_info: Optional[Dict] = None, gemini_key: str = ""):
        self.df = df
        self.schema_info = schema_info or {}
        self.gemini_key = gemini_key
        self.history: List[Dict[str, str]] = []
        self.model = None
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except ImportError:
                self.model = None

    def ask(self, question: str) -> str:
        """Asks a question about the dataset grounded in schema and summary statistics."""
        if not self.gemini_key:
            return "Please provide a valid Gemini API key to use conversational chat."
        if not self.model:
            return "google-generativeai package is not installed."

        context = f"""
Dataset Summary:
- Rows: {len(self.df)} | Columns: {self.df.columns.tolist()}
- Dtypes: {self.df.dtypes.to_dict()}
- Sample Head:
{self.df.head(5).to_string()}
"""
        prompt = f"""
System Context: You are InsightForge AI assistant. Answer user questions accurately based on the dataset.
{context}

Previous Conversation History:
{self.history[-4:] if self.history else 'None'}

User Question: {question}
"""
        response = self.model.generate_content(prompt)
        ans = response.text.strip()
        self.history.append({"user": question, "assistant": ans})
        return ans

    def reset(self):
        """Clears chat history."""
        self.history = []

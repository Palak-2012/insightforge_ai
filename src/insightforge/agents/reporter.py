"""
InsightForge AI — Agent 6: Report Agent
=======================================
Generates a structured, professional executive PDF report from all pipeline outputs using fpdf2.
Includes fallback text report generation if fpdf2 is not yet installed.
"""

import os
from typing import Any, Dict

from insightforge.state import InsightForgeState
from insightforge.logger import log_event

try:
    from fpdf import FPDF

    class InsightReport(FPDF):
        """Custom styled PDF document for InsightForge AI."""

        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(33, 37, 41)
            self.cell(0, 10, "InsightForge AI — Automated Business Report", align="C", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(108, 117, 125)
            self.cell(0, 5, "Multi-Agent Data Intelligence & Strategic Insights", align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Generated with InsightForge AI", align="C")

        def section_title(self, title: str):
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(13, 110, 253)
            self.ln(4)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(13, 110, 253)
            self.set_line_width(0.4)
            self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
            self.ln(3)

        def write_body(self, text: str):
            self.set_font("Helvetica", "", 10)
            self.set_text_color(40, 40, 40)
            clean_text = text.encode("latin-1", "replace").decode("latin-1")
            self.multi_cell(0, 6, clean_text)
            self.ln(2)

except ImportError:
    InsightReport = None


def generate_pdf_report(state: Dict[str, Any], output_path: str = "insightforge_report.pdf") -> str:
    """Compiles the full state into a PDF report (or text report fallback)."""
    schema = state.get("schema_info", {})
    cleaning = state.get("cleaning_report", {})
    insights = state.get("insights", "No insights generated.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if InsightReport is not None:
        pdf = InsightReport()
        pdf.alias_nb_pages()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # 1. Executive Summary & Schema
        pdf.section_title("1. Dataset & Schema Overview")
        overview_text = (
            f"Domain: {schema.get('domain', 'Generic')}\n"
            f"Total Records: {schema.get('total_rows', 0):,} rows | {schema.get('total_columns', 0)} columns\n"
            f"Target Variable: {schema.get('target_variable', 'None detected')}\n"
            f"Numeric Features: {schema.get('numeric_count', 0)} | Categorical Features: {schema.get('categorical_count', 0)}\n"
        )
        pdf.write_body(overview_text)

        # 2. Data Cleaning
        pdf.section_title("2. Data Quality & Cleaning Actions")
        cleaning_text = (
            f"Initial Rows: {cleaning.get('initial_rows', 0):,} -> Final Rows: {cleaning.get('final_rows', 0):,}\n"
            f"Duplicate Records Removed: {cleaning.get('duplicates_removed', 0)}\n"
            f"Missing Values Imputed: {cleaning.get('nulls_filled_total', 0)}\n"
            f"Outliers Identified (IQR): {cleaning.get('total_outliers', 0)}\n"
        )
        pdf.write_body(cleaning_text)

        # 3. AI Insights
        pdf.section_title("3. Strategic Business Insights & Recommendations")
        pdf.write_body(insights)

        pdf.output(output_path)
    else:
        # Fallback text output if fpdf2 is not installed
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("InsightForge AI — Automated Business Report\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. DATASET OVERVIEW\n")
            f.write(f"Domain: {schema.get('domain')}\n")
            f.write(f"Rows: {schema.get('total_rows')} | Cols: {schema.get('total_columns')}\n\n")
            f.write("2. DATA CLEANING\n")
            f.write(f"Duplicates: {cleaning.get('duplicates_removed')}\n")
            f.write(f"Nulls Filled: {cleaning.get('nulls_filled_total')}\n\n")
            f.write("3. INSIGHTS & RECOMMENDATIONS\n")
            f.write(insights + "\n")
        output_path = txt_path

    return output_path


def report_agent(state: InsightForgeState) -> Dict[str, Any]:
    """
    Agent 6 — Report Agent
    ----------------------
    Reads  : All state fields
    Writes : state["pdf_path"], state["pipeline_log"]
    """
    state["pipeline_log"] = log_event(state, "report_agent", "Assembling executive report...")
    output_path = state.get("pdf_path") or "insightforge_report.pdf"

    try:
        saved_path = generate_pdf_report(state, output_path)
        state["pdf_path"] = saved_path
        state["pipeline_log"] = log_event(state, "report_agent", f"Report saved at: {saved_path}")
    except Exception as e:
        state["errors"].append(f"Report Agent Error: {str(e)}")
        state["pdf_path"] = ""

    return state

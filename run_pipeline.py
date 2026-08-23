"""
InsightForge AI — Command Line Interface Runner
================================================
Executes the full 7-agent pipeline on any input dataset.

Usage:
    python run_pipeline.py --data data/titanic.csv --output reports/titanic_report.pdf
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from insightforge.agents.supervisor import run_pipeline
from insightforge.advanced.anomaly_detection import detect_anomalies


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="InsightForge AI — Automated Multi-Agent Data Analysis Pipeline"
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="data/titanic.csv",
        help="Path to the input CSV or Excel dataset"
    )
    parser.add_argument(
        "--gemini-key",
        "-k",
        type=str,
        default=os.getenv("GEMINI_API_KEY", ""),
        help="Google Gemini API key (or set GEMINI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/insightforge_report.pdf",
        help="Output path for the generated PDF executive report"
    )
    parser.add_argument(
        "--anomalies",
        action="store_true",
        help="Run explicit IQR anomaly detection"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("       InsightForge AI — Automated Data Science Copilot")
    print("=" * 65)
    print(f"  Dataset Path : {args.data}")
    print(f"  Output Report: {args.output}")
    print(f"  Gemini Key   : {'[SET]' if args.gemini_key else '[UNSET - Using rule-based fallback]'}")
    print("=" * 65)

    try:
        final_state = run_pipeline(
            data_source=args.data,
            gemini_key=args.gemini_key,
            output_pdf=args.output
        )

        print("\n" + "=" * 65)
        print("  PIPELINE EXECUTION SUMMARY")
        print("=" * 65)
        print(f"  Domain Detected     : {final_state['schema_info'].get('domain', 'N/A')}")
        print(f"  Records Processed   : {len(final_state['cleaned_df']):,} rows | {len(final_state['cleaned_df'].columns)} cols")
        print(f"  Duplicates Removed  : {final_state['cleaning_report'].get('duplicates_removed', 0)}")
        print(f"  Null Values Filled  : {final_state['cleaning_report'].get('nulls_filled_total', 0)}")
        print(f"  Plotly Charts Built : {len(final_state.get('charts', []))}")
        print(f"  Report File Saved   : {final_state.get('pdf_path', 'N/A')}")

        if args.anomalies and final_state.get("cleaned_df") is not None:
            print("\n  ANOMALY DETECTION (IQR):")
            anomalies = detect_anomalies(final_state["cleaned_df"])
            for col, ainfo in anomalies.items():
                print(f"    - {col}: {ainfo['outlier_count']} outliers ({ainfo['outlier_pct']}%)")

        print("=" * 65)
        print("\nINSIGHTS PREVIEW:")
        print("-" * 65)
        print(final_state.get("insights", "")[:600] + "...")
        print("-" * 65)

    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import os
import pandas as pd
from insightforge.state import create_initial_state
from insightforge.agents.supervisor import run_pipeline


def test_create_initial_state():
    state = create_initial_state(dataset_path="sample.csv")
    assert state["dataset_path"] == "sample.csv"
    assert state["raw_df"] is None
    assert state["cleaned_df"] is None
    assert isinstance(state["errors"], list)
    assert isinstance(state["pipeline_log"], list)


def test_full_pipeline_offline(tmp_path):
    # Test full 7-agent pipeline offline without Gemini API key
    df = pd.DataFrame({
        "Age": [22, 38, 26, 35, 54, None],
        "Fare": [7.25, 71.83, 7.92, 53.1, 8.05, 1000.0],
        "Survived": [0, 1, 1, 1, 0, 1]
    })
    pdf_out = str(tmp_path / "test_report.pdf")
    final_state = run_pipeline(data_source=df, gemini_key="", output_pdf=pdf_out)

    assert final_state["cleaned_df"] is not None
    assert len(final_state["cleaned_df"]) == 6
    assert final_state["cleaning_report"]["nulls_filled_total"] == 1
    assert len(final_state["charts"]) > 0
    assert os.path.exists(final_state["pdf_path"])
    assert "Executive Summary" in final_state["insights"]

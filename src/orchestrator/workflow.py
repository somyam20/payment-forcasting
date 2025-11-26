import os
import json
import logging
from typing import Optional, Dict, Any

from src.llm.litellm_client import lite_client
from src.llm.prompt_loader import prompt_loader
from src.data_pipeline.s3_handler import upload_file, download_file
from src.data_pipeline.file_processor import (
    load_dataframe,
    classify,
    sample_rows,
)
from src.data_pipeline.stats_generator import get_descriptive_stats
from src.data_pipeline.time_series import run_lstm_forecast

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


class WorkflowEngine:
    """
    End-to-end pipeline for dataset analysis:
    - Saves file to S3
    - Loads DataFrame
    - Classifies small vs large
    - Generates stats
    - Runs LLM forecasting (small files)
    - Runs LSTM model (large files) then LLM analysis
    """

    def __init__(self):
        self.temp_dir = os.getenv("TEMP_DIR", "data/temp")
        os.makedirs(self.temp_dir, exist_ok=True)

    # -------------------------------------------------------------
    # MAIN ENTRYPOINT
    # -------------------------------------------------------------
    def run(
        self,
        local_path: str,
        s3_key: str,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core flow:
        1. Upload to S3
        2. Re-download locally (clean separation)
        3. Load DF
        4. Classify (small/large)
        5. Generate stats
        6. Small file: LLM prediction
        7. Large file: LSTM forecast + LLM explanation

        Returns clean JSON response.
        """

        logger.info(f"Starting workflow for file: {local_path}")

        # 1. Upload original file to S3
        s3_uri = upload_file(local_path, s3_key)

        # 2. Download temp copy for processing
        local_tmp = download_file(s3_key, os.path.join(self.temp_dir, s3_key))

        # 3. Load DataFrame
        df = load_dataframe(local_tmp)

        # 4. Small or large file?
        size_type = classify(df)

        # 5. Stats
        stats = get_descriptive_stats(df)

        # branching
        if size_type == "small":
            result = self._handle_small_file(df, stats)
        else:
            result = self._handle_large_file(df, stats, target_column)

        # attach s3 uri
        result["s3_uri"] = s3_uri
        result["file_size"] = size_type

        return result

    # -------------------------------------------------------------
    # SMALL FILE HANDLER (LLM FORECASTING)
    # -------------------------------------------------------------
    def _handle_small_file(self, df, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        For <= 2000 rows:
        - Take descriptive stats
        - Provide sample rows
        - LLM generates forecasts/trends/insights
        """

        sample = sample_rows(df, n=50)

        prompt = prompt_loader.get("small_file_analysis")
        prompt = (
            prompt.replace("{{row_count}}", str(len(df)))
                  .replace("{{stats_json}}", json.dumps(stats, indent=2))
                  .replace("{{sample_rows}}", json.dumps(sample, indent=2))
        )

        analysis = lite_client.chat(prompt)

        return {
            "stats": stats,
            "forecast_from_llm": analysis
        }

    # -------------------------------------------------------------
    # LARGE FILE HANDLER (LSTM FORECAST + LLM ANALYSIS)
    # -------------------------------------------------------------
    def _handle_large_file(
        self,
        df,
        stats: Dict[str, Any],
        target_column: Optional[str]
    ) -> Dict[str, Any]:
        """
        For > 2000 rows:
        - Must specify target column for numeric forecasting
        - LSTM produces numeric forecast
        - LLM generates narrative analysis
        """

        if not target_column:
            raise ValueError(
                "target_column is required for large file forecasting"
            )

        # numeric prediction
        forecast_numeric = run_lstm_forecast(df, target_column)

        prompt = prompt_loader.get("large_file_analysis")
        prompt = (
            prompt.replace("{{row_count}}", str(len(df)))
                  .replace("{{stats_json}}", json.dumps(stats, indent=2))
                  .replace("{{forecast_values}}", json.dumps(forecast_numeric, indent=2))
                  .replace("{{target_column}}", target_column)
        )

        narrative = lite_client.chat(prompt)

        return {
            "stats": stats,
            "forecast_lstm": forecast_numeric,
            "analysis_from_llm": narrative
        }


# Singleton export
workflow_engine = WorkflowEngine()

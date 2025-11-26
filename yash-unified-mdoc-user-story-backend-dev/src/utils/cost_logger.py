from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

@dataclass
class CostLoggerConfig:
    username: Optional[str] = None
    time: Optional[str] = None
    guid: Optional[str] = None
    document_type: Optional[str] = None

    client_name: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[str] = None
    video_duration: Optional[str] = None
    screenshot_processing: Optional[str] = None
    document_generation: Optional[str] = None
    prompt_tokens: Optional[str] = None
    completion_tokens: Optional[str] = None
    total_tokens: Optional[str] = None
    open_ai_cost: Optional[str] = None
    azure_whisper_cost: Optional[str] = None
    total_cost: Optional[str] = None

class UsageCostLogger:
    def __init__(self):
        self.config = CostLoggerConfig()
        # Load configuration from YAML
        try:
            from .config_loader import get_config_loader
            config_loader = get_config_loader()
            app_config = config_loader.get_config('app_config.yaml')
            storage_config = app_config.get('storage', {})
            
            # Get the full path from config, or construct it
            usage_cost_log_file = storage_config.get('usage_cost_log_file', 'usage_cost_log.csv')
            if os.path.isabs(usage_cost_log_file) or os.path.dirname(usage_cost_log_file):
                # Full path provided
                self.CSV_FILE_PATH = usage_cost_log_file
                self.LOCAL_STORAGE_DIR = os.path.dirname(usage_cost_log_file)
            else:
                # Just filename, use with storage dir
                self.LOCAL_STORAGE_DIR = storage_config.get('local_storage_dir', 'data/outputs')
                self.CSV_FILE_PATH = os.path.join(self.LOCAL_STORAGE_DIR, usage_cost_log_file)
        except Exception:
            # Fallback to environment variables or defaults
            self.LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR") or "data/outputs"
            csv_filename = os.getenv("USAGE_COST_BLOB_NAME") or "usage_cost_log.csv"
            self.CSV_FILE_PATH = os.path.join(self.LOCAL_STORAGE_DIR, csv_filename)
        
        # Create local storage directory if it doesn't exist
        os.makedirs(self.LOCAL_STORAGE_DIR, exist_ok=True)

    def get_user_info(self):
        headers = st.context.headers
        user_name = headers.get("X-Ms-Client-Principal-Name", "user")
        user_email = headers.get("X-Ms-Client-Principal-Email", "email")
        return {"name": user_name, "email": user_email}

    def load_csv_from_local(self):
        """Load CSV from local file, create if it doesn't exist"""
        if os.path.exists(self.CSV_FILE_PATH):
            try:
                return pd.read_csv(self.CSV_FILE_PATH)
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                # Return empty dataframe with correct columns
                return self._create_empty_dataframe()
        else:
            # Create empty dataframe with headers
            return self._create_empty_dataframe()
    
    def _create_empty_dataframe(self):
        """Create an empty dataframe with the correct column structure"""
        return pd.DataFrame(columns=[
            "username", "time", "guid", "document_type", "client_name",
            "file_name", "file_size", "video_duration", "screenshot_processing",
            "document_generation", "prompt_tokens", "completion_tokens",
            "total_tokens", "open_ai_cost", "azure_whisper_cost", "total_cost"
        ])

    def add_row(self, dataframe):
        row = pd.DataFrame({
            "username": [self.config.username],
            "time": [self.config.time],
            "guid": [self.config.guid],
            "document_type": [self.config.document_type],
            "client_name": [self.config.client_name],
            "file_name": [self.config.file_name],
            "file_size": [self.config.file_size],
            "video_duration": [self.config.video_duration],
            "screenshot_processing": [self.config.screenshot_processing],
            "document_generation": [self.config.document_generation],
            "prompt_tokens": [self.config.prompt_tokens],
            "completion_tokens": [self.config.completion_tokens],
            "total_tokens": [self.config.total_tokens],
            "open_ai_cost": [self.config.open_ai_cost],
            "azure_whisper_cost": [self.config.azure_whisper_cost],
            "total_cost": [self.config.total_cost],
        })
        # Suppress FutureWarning for empty DataFrames
        if len(dataframe) == 0:
            return row
        return pd.concat([dataframe, row], ignore_index=True)

    def save_csv_to_local(self, dataframe):
        """Save CSV to local file"""
        try:
            dataframe.to_csv(self.CSV_FILE_PATH, index=False)
            print("Saved Successfully to local storage")
        except Exception as e:
            print(f"Error saving to local file: {e}")

    def log(self, **kwargs):
        user_data = self.get_user_info()
        self.config.username = user_data["name"]
        self.config.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.config.client_name = kwargs["client_name"]
        self.config.guid = kwargs["guid"]
        self.config.document_type = kwargs.get("document_type")

        # Optional fields
        self.config.file_name = kwargs.get("file_name")
        self.config.file_size = kwargs.get("file_size")
        self.config.video_duration = kwargs.get("video_duration")
        self.config.screenshot_processing = kwargs.get("screenshot_processing")
        self.config.document_generation = kwargs.get("document_generation")
        self.config.prompt_tokens = kwargs.get("prompt_tokens")
        self.config.completion_tokens = kwargs.get("completion_tokens")
        self.config.total_tokens = kwargs.get("total_tokens")
        self.config.open_ai_cost = kwargs.get("open_ai_cost")
        self.config.azure_whisper_cost = kwargs.get("azure_whisper_cost")
        self.config.total_cost = kwargs.get("total_cost")

        df = self.load_csv_from_local()
        df = self.add_row(df)
        self.save_csv_to_local(df)

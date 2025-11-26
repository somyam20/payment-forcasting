from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

@dataclass
class LoggerConfig:
    username: Optional[str] = None
    log_in_time: Optional[str] = None
    client_name: Optional[str] = None
    tool_name: Optional[str] = None
    email: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[str] = None
    record_count: Optional[str] = None
    file_type: Optional[str] = None

class Logger:
    def __init__(self):
        self.config = LoggerConfig()
        # Load configuration from YAML
        try:
            from .config_loader import get_config_loader
            config_loader = get_config_loader()
            app_config = config_loader.get_config('app_config.yaml')
            storage_config = app_config.get('storage', {})
            
            # Get the full path from config, or construct it
            audit_log_file = storage_config.get('audit_log_file', 'audit_log.csv')
            if os.path.isabs(audit_log_file) or os.path.dirname(audit_log_file):
                # Full path provided
                self.CSV_FILE_PATH = audit_log_file
                self.LOCAL_STORAGE_DIR = os.path.dirname(audit_log_file)
            else:
                # Just filename, use with storage dir
                self.LOCAL_STORAGE_DIR = storage_config.get('local_storage_dir', 'data/outputs')
                self.CSV_FILE_PATH = os.path.join(self.LOCAL_STORAGE_DIR, audit_log_file)
        except Exception:
            # Fallback to environment variables or defaults
            self.LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR") or "data/outputs"
            csv_filename = os.getenv("BLOB_NAME") or "audit_log.csv"
            self.CSV_FILE_PATH = os.path.join(self.LOCAL_STORAGE_DIR, csv_filename)
        os.makedirs(self.LOCAL_STORAGE_DIR, exist_ok=True)
        print("Container_Name", os.getenv("CONTAINER_NAME"))

    def get_user_info(self):
        headers = st.context.headers
        if "X-Ms-Client-Principal-Name" in headers:
            user_name = headers["X-Ms-Client-Principal-Name"]
        else:
            user_name= 'user'
        if "X-Ms-Client-Principal-Email" in headers:
            user_email = headers["X-Ms-Client-Principal-Email"]
        else:
            user_email = 'email'
        return {
            "name": user_name,
            "email": user_email
        }            

    def load_csv_from_local(self):
        """Load CSV from local file, create empty dataframe if not exists or unreadable."""
        if os.path.exists(self.CSV_FILE_PATH):
            try:
                return pd.read_csv(self.CSV_FILE_PATH)
            except Exception:
                pass
        return pd.DataFrame({
            "username": pd.Series(dtype="str"),
            "email": pd.Series(dtype="str"),
            "client_name": pd.Series(dtype="str"),
            "login_time": pd.Series(dtype="str"),
            "tool_name": pd.Series(dtype="str"),
            "file_name": pd.Series(dtype="str"),
            "file_size": pd.Series(dtype="str"),
            "file_type": pd.Series(dtype="str"),
            "record_count": pd.Series(dtype="str"),
        })
    
    def add_row(self, dataframe):
        row_dataframe = pd.DataFrame({
            "username": [self.config.username],
            "email": [self.config.email],
            "client_name": [self.config.client_name],
            "login_time": [self.config.log_in_time],
            "tool_name": [self.config.tool_name],
            "file_name": [self.config.file_name],
            "file_size": [self.config.file_size],
            "file_type": [self.config.file_type],
            "record_count": [self.config.record_count]
        })
        return pd.concat([dataframe, row_dataframe])
        
    def save_csv_to_local(self, dataframe):
        try:
            dataframe.to_csv(self.CSV_FILE_PATH, index=False)
            print("Saved Successfully to local storage")
        except Exception as e:
            print(f"Error saving audit log locally: {e}")

    def log(self, **kwargs):
        """
        The user has to pass the client_name and tool_name as mandatory parameters.
        Few other optional parameters are:
        1. file_name
        2. file_size
        3. file_type
        4. record_count
        """
        # Mandatory fields
        user_data = self.get_user_info()
        self.config.username = user_data["name"]
        self.config.email = user_data["email"]
        self.config.client_name = kwargs["client_name"]
        self.config.tool_name = kwargs["tool_name"]
        self.config.log_in_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Optional fields
        self.config.file_name = kwargs.get("file_name")
        self.config.file_size = kwargs.get("file_size")
        self.config.record_count = kwargs.get("record_count")
        self.config.file_type = kwargs.get("file_type")

        # Loading CSV locally
        dataframe = self.load_csv_from_local()

        # Add row to dataframe and get updated dataframe
        dataframe = self.add_row(dataframe)

        # Save CSV locally
        self.save_csv_to_local(dataframe)

    def usage_log(self, **kwargs):
        """
        The user has to pass the client_name and tool_name as mandatory parameters.
        Few other optional parameters are:
        1. file_name
        2. file_size
        3. file_type
        4. record_count
        """
        # Mandatory fields
        user_data = self.get_user_info()
        self.config.username = user_data["name"]
        self.config.email = user_data["email"]
        self.config.client_name = kwargs["client_name"]
        self.config.tool_name = kwargs["tool_name"]
        self.config.log_in_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Optional fields
        self.config.file_name = kwargs.get("file_name")
        self.config.file_size = kwargs.get("file_size")
        self.config.record_count = kwargs.get("record_count")
        self.config.file_type = kwargs.get("file_type")

        # Loading CSV locally
        dataframe = self.load_csv_from_local()

        # Add row to dataframe and get updated dataframe
        dataframe = self.add_row(dataframe)

        # Save CSV locally
        self.save_csv_to_local(dataframe)
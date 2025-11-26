"""
Configuration module for OpenAI and Azure OpenAI APIs
This module handles both standard OpenAI and Azure OpenAI authentication
"""

import os
from typing import Optional
from openai import OpenAI, AzureOpenAI

from dotenv import load_dotenv
import os
import logging

from .logger_config import setup_logger
load_dotenv()

setup_logger()

# Load the .env file
load_dotenv()

# Replace with your Key Vault URL
key_vault_url = os.getenv('KEY_VAULT_URL')

# Try to use Key Vault if configured, otherwise fall back to environment variables
if key_vault_url:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient
        
        credential = DefaultAzureCredential()
        vault_client = SecretClient(vault_url=key_vault_url, credential=credential)
        
        # Environment variables from Key Vault
        OPENAI_API_KEY = vault_client.get_secret("OPENAI-API-KEY").value
        AZURE_OPENAI_API_KEY = vault_client.get_secret("OPENAI-API-KEY").value
        AZURE_OPENAI_ENDPOINT = vault_client.get_secret("OPENAI-API-ENDPOINT").value
        AZURE_OPENAI_API_VERSION = vault_client.get_secret("OPENAI-API-VERSION").value
        AZURE_GPT_DEPLOYMENT_NAME = vault_client.get_secret("GPT-DEPLOYMENT-NAME").value
    except Exception as e:
        logging.warning(f"Key Vault access failed, falling back to environment variables: {e}")
        # Fall back to environment variables
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        AZURE_GPT_DEPLOYMENT_NAME = os.getenv("AZURE_GPT_DEPLOYMENT_NAME", "")
else:
    # Use environment variables directly (bypass Key Vault)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_GPT_DEPLOYMENT_NAME = os.getenv("AZURE_GPT_DEPLOYMENT_NAME", "") 

# Provide Litellm-compatible environment variable aliases so both naming schemes work
if AZURE_OPENAI_API_KEY and not os.getenv("AZURE_API_KEY"):
    os.environ["AZURE_API_KEY"] = AZURE_OPENAI_API_KEY
if AZURE_OPENAI_ENDPOINT and not os.getenv("AZURE_API_BASE"):
    os.environ["AZURE_API_BASE"] = AZURE_OPENAI_ENDPOINT
if AZURE_OPENAI_API_VERSION and not os.getenv("AZURE_API_VERSION"):
    os.environ["AZURE_API_VERSION"] = AZURE_OPENAI_API_VERSION
if AZURE_GPT_DEPLOYMENT_NAME and not os.getenv("AZURE_API_DEPLOYMENT_NAME"):
    os.environ["AZURE_API_DEPLOYMENT_NAME"] = AZURE_GPT_DEPLOYMENT_NAME

# Check if Azure OpenAI or regular OpenAI should be used
USE_AZURE = AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_GPT_DEPLOYMENT_NAME

# Check if API keys are available
OPENAI_AVAILABLE = (USE_AZURE and AZURE_OPENAI_API_KEY) or (not USE_AZURE and OPENAI_API_KEY)

def get_openai_client() -> Optional[OpenAI | AzureOpenAI]:
    """
    Get the appropriate OpenAI client based on available credentials.
    Will try Azure OpenAI first, then fall back to standard OpenAI.
    
    Returns:
        OpenAI or AzureOpenAI client, or None if no credentials are available
    """
    # Check for Azure OpenAI credentials
    if USE_AZURE:
        try:
            logging.info(f"Initializing Azure OpenAI client with endpoint: {AZURE_OPENAI_ENDPOINT}")
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            return client
        except Exception as e:
            logging.exception(f"Failed to initialize Azure OpenAI client: {e}")
    
    # Fall back to standard OpenAI
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            return client
        except Exception as e:
            logging.exception(f"Failed to initialize standard OpenAI client: {e}")
    
    return None

def get_chat_model_name() -> str:
    """
    Get the appropriate model name/deployment for chat completions
    
    Returns:
        Model name for standard OpenAI or deployment name for Azure OpenAI
    """
    if USE_AZURE:
        return AZURE_GPT_DEPLOYMENT_NAME
    else:
        return "gpt-4o"  # Default to gpt-4o for standard OpenAI
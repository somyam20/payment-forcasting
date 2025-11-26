#!/usr/bin/env python3
"""
Setup script for Azure OpenAI credentials
This script helps set up environment variables for Azure OpenAI Service.
"""

import os
import sys
import argparse
from litellm import completion
import litellm

import subprocess


def main():
    parser = argparse.ArgumentParser(description="Set up Azure OpenAI credentials")
    parser.add_argument("--key", help="Azure OpenAI API Key")
    parser.add_argument("--endpoint", help="Azure OpenAI Endpoint URL")
    parser.add_argument("--api-version", default="2024-05-01-preview", help="API Version (default: 2024-05-01-preview)")
    parser.add_argument("--deployment", default="gpt-4o", help="Model deployment name (default: gpt-4o)")
    
    args = parser.parse_args()
    
    # Interactive mode if no arguments provided
    if len(sys.argv) == 1:
        print("Azure OpenAI Setup Utility")
        print("=========================")
        print("This script will help you set up Azure OpenAI credentials.")
        print()
        
        key = input("Enter your Azure OpenAI API Key: ").strip()
        endpoint = input("Enter your Azure OpenAI Endpoint URL: ").strip()
        api_version = input("Enter API Version [2024-05-01-preview]: ").strip() or "2024-05-01-preview"
        deployment = input("Enter Model Deployment Name [gpt-4o]: ").strip() or "gpt-4o"
    else:
        key = args.key
        endpoint = args.endpoint
        api_version = args.api_version
        deployment = args.deployment
    
    # Validate inputs
    if not key:
        print("Error: API Key is required")
        return 1
    
    if not endpoint:
        print("Error: Endpoint URL is required")
        return 1
    
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_API_ENDPOINT"] = endpoint
    os.environ["OPENAI_API_VERSION"] = api_version
    os.environ["GPT_DEPLOYMENT_NAME"] = deployment
    
    # Save to .env file for persistence
    with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={key}\n")
        f.write(f"OPENAI_API_ENDPOINT={endpoint}\n")
        f.write(f"OPENAI_API_VERSION={api_version}\n")
        f.write(f"GPT_DEPLOYMENT_NAME={deployment}\n")
    
    print("\nAzure OpenAI credentials have been set!")
    print("Environment variables are now available for your application.")
    print("These settings have also been saved to the .env file for future use.")
    
    # Ask if the user wants to test the connection
    test = input("\nDo you want to test the connection? (y/n): ").strip().lower()
    if test == 'y':
        print("\nTesting connection to Azure OpenAI...")
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            # Prepare messages according to OpenAI's typing requirements
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! This is a test of the Azure OpenAI connection."}
            ]
            
            try:
                print(f"Testing Azure OpenAI with model/deployment: {deployment}")
                # Azure OpenAI requires BOTH model and deployment_id parameters
                response = completion(
                    model=f"azure/{deployment}",      # Required for Azure!
                    # deployment_id=deployment,
                    messages=messages,
                    max_tokens=100
                )
                print("API call successful")
            except Exception as e:
                print(f"API call error details: {str(e)}")
                raise
            
            print("\nConnection successful! Response:")
            print("-------------------------------")
            print(response.choices[0].message.content)
            print("-------------------------------")
            print("\nYour Azure OpenAI credentials are working correctly!")
            
        except Exception as e:
            print(f"\nError testing connection: {e}")
            print("\nPlease check your credentials and try again.")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
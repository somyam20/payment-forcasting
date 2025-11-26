import json
import os
from dotenv import load_dotenv

load_dotenv()

def extract_token_usage_from_app_log(log_file_path="usage.log", session_id=None):
    """
    Extract token usage from app.log, optionally filtered by session_id.
    
    Args:
        log_file_path: Path to your app.log file
        session_id: Optional session ID to filter usage for
        
    Returns:
        dict: Contains prompt_tokens, completion_tokens, total_tokens, openai_cost
    """
    input_cost = float(os.getenv("INPUT_TOKEN_COST", 0))
    output_cost = float(os.getenv("OUTPUT_TOKEN_COST", 0))
    
    if not os.path.exists(log_file_path):
        print(f"❌ Log file not found: {log_file_path}")
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "openai_cost": 0.0
        }
    
    openai_entries = []
    
    # Read and parse the log file
    with open(log_file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Look for OpenAI usage entries
                if "OPENAI_USAGE:" in line:
                    json_start = line.find('{"timestamp"')
                    if json_start != -1:
                        json_str = line[json_start:].strip()
                        entry = json.loads(json_str)
                        
                        # Filter by session_id if provided
                        if session_id is None or entry.get("session_id") == session_id:
                            openai_entries.append(entry)
                            
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Could not parse JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Warning: Error processing line {line_num}: {e}")
                continue
    
    # Calculate totals
    total_prompt_tokens = sum(entry.get("prompt_tokens", 0) for entry in openai_entries)
    total_completion_tokens = sum(entry.get("completion_tokens", 0) for entry in openai_entries)
    total_tokens = sum(entry.get("total_tokens", 0) for entry in openai_entries)
    
    # Calculate cost
    openai_cost = (total_prompt_tokens * input_cost + total_completion_tokens * output_cost) / 100000
    
    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "openai_cost": round(openai_cost, 4)
    }
import json
import os
import logging
from datetime import datetime
from typing import Optional
from .logger_config import setup_usage_logger
import streamlit as st
# Use existing log configuration - don't create new files
AGGREGATE_FILE = "token_aggregate.json"

usage_logger = setup_usage_logger()


def _resolve_session_id(explicit_session_id: Optional[str] = None) -> str:
    """
    Determine the session identifier for usage logging without assuming a Streamlit context.
    """
    if explicit_session_id:
        return explicit_session_id

    # Try Streamlit session state if available
    try:
        session_state = getattr(st, "session_state", None)
        if session_state is not None:
            # Prefer dict-like access to avoid AttributeError when key is absent
            if isinstance(session_state, dict):
                session_id = session_state.get("session_guid")
            else:
                session_id = session_state.get("session_guid") if hasattr(session_state, "get") else None
                if session_id is None and hasattr(session_state, "session_guid"):
                    session_id = session_state.session_guid
            if session_id:
                return str(session_id)
    except Exception:
        pass

    # Fall back to environment variable or default placeholder
    env_session = os.getenv("SESSION_GUID")
    if env_session:
        return env_session

    return "unknown-session"


def log_openai_usage(module_name: str, model: str, prompt_tokens: int, completion_tokens: int, function_name: str = "", session_id: Optional[str] = None):
    """
    Log OpenAI API usage using the existing logger configuration.
    
    Args:
        module_name: Name of the module/script using the API
        model: Model name used
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens  
        function_name: Specific function that made the call
    """
    total_tokens = prompt_tokens + completion_tokens
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "service": "openai",
        "session_id": _resolve_session_id(session_id),
        "module": module_name,
        "function": function_name,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
    
    # Use existing logger instead of creating new log file
    usage_logger.info(f"OPENAI_USAGE: {json.dumps(entry)}")
    _update_aggregate("openai", module_name, function_name, total_tokens)

def log_whisper_usage(module_name: str, model: str, duration_seconds: float, function_name: str = "", file_size_mb: Optional[float] = None):
    """
    Log Whisper API usage using the existing logger configuration.
    
    Args:
        module_name: Name of the module/script using the API
        model: Model name used (whisper-1)
        duration_seconds: Audio duration in seconds
        function_name: Specific function that made the call
        file_size_mb: Optional file size in MB
    """
    duration_minutes = duration_seconds / 60
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "service": "whisper",
        "module": module_name,
        "function": function_name,
        "model": model,
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
        "file_size_mb": file_size_mb
    }
    
    # Use existing logger instead of creating new log file
    usage_logger.info(f"WHISPER_USAGE: {json.dumps(entry)}")
    _update_aggregate("whisper", module_name, function_name, duration_minutes)

def _update_aggregate(service: str, module_name: str, function_name: str, usage: float):
    """
    Update aggregate usage statistics.
    
    Args:
        service: Service name (openai/whisper)
        module_name: Module name
        function_name: Function name
        usage: Usage amount (tokens for OpenAI, minutes for Whisper)
    """
    try:
        # Load existing aggregate data
        if os.path.exists(AGGREGATE_FILE):
            with open(AGGREGATE_FILE, "r") as f:
                aggregate = json.load(f)
        else:
            aggregate = {
                "openai": {"total": 0, "by_module": {}, "by_function": {}},
                "whisper": {"total": 0, "by_module": {}, "by_function": {}}
            }

        # Ensure service exists in aggregate
        if service not in aggregate:
            aggregate[service] = {"total": 0, "by_module": {}, "by_function": {}}

        # Update total usage
        aggregate[service]["total"] += usage
        
        # Update by module
        if module_name not in aggregate[service]["by_module"]:
            aggregate[service]["by_module"][module_name] = 0
        aggregate[service]["by_module"][module_name] += usage
        
        # Update by function (if provided)
        if function_name:
            function_key = f"{module_name}.{function_name}"
            if function_key not in aggregate[service]["by_function"]:
                aggregate[service]["by_function"][function_key] = 0
            aggregate[service]["by_function"][function_key] += usage

        # Save updated aggregate
        with open(AGGREGATE_FILE, "w") as f:
            json.dump(aggregate, f, indent=2)
            
    except Exception as e:
        usage_logger.error(f"Error updating token usage aggregate: {e}")

def get_usage_summary():
    """
    Get a summary of all API usage.
    
    Returns:
        dict: Usage summary data
    """
    if not os.path.exists(AGGREGATE_FILE):
        return {"openai": {"total": 0}, "whisper": {"total": 0}}
    
    try:
        with open(AGGREGATE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        usage_logger.error(f"Error reading usage summary: {e}")
        return {"openai": {"total": 0}, "whisper": {"total": 0}}

def print_usage_report():
    """Print a formatted usage report."""
    summary = get_usage_summary()
    
    print("\n" + "="*50)
    print("API USAGE REPORT")
    print("n="*50)
    
    # OpenAI Usage
    if "openai" in summary and summary["openai"]["total"] > 0:
        print(f"\n🤖 OpenAI Usage:")
        print(f"   Total Tokens: {summary['openai']['total']:,}")
        
        if "by_module" in summary["openai"]:
            print(f"   By Module:")
            for module, tokens in summary["openai"]["by_module"].items():
                print(f"     {module}: {tokens:,} tokens")
        
        if "by_function" in summary["openai"]:
            print(f"   By Function:")
            for func, tokens in summary["openai"]["by_function"].items():
                print(f"     {func}: {tokens:,} tokens")
    else:
        print(f"\n🤖 OpenAI Usage: No usage recorded")
    
    # Whisper Usage  
    if "whisper" in summary and summary["whisper"]["total"] > 0:
        print(f"\n🎤 Whisper Usage:")
        print(f"   Total Minutes: {summary['whisper']['total']:.2f}")
        
        if "by_module" in summary["whisper"]:
            print(f"   By Module:")
            for module, minutes in summary["whisper"]["by_module"].items():
                print(f"     {module}: {minutes:.2f} minutes")
        
        if "by_function" in summary["whisper"]:
            print(f"   By Function:")
            for func, minutes in summary["whisper"]["by_function"].items():
                print(f"     {func}: {minutes:.2f} minutes")
    else:
        print(f"\n🎤 Whisper Usage: No usage recorded")
    
    print("="*50)

def parse_usage_from_app_log(log_file_path: str = "app.log"):
    """
    Parse token usage from your existing app.log file.
    
    Args:
        log_file_path: Path to your app.log file
        
    Returns:
        List of usage entries
    """
    usage_entries = []
    
    if not os.path.exists(log_file_path):
        usage_logger.warning(f"Log file not found: {log_file_path}")
        return usage_entries
    
    try:
        with open(log_file_path, "r") as f:
            for line in f:
                if "OPENAI_USAGE:" in line or "WHISPER_USAGE:" in line:
                    try:
                        # Extract JSON part
                        if "OPENAI_USAGE:" in line:
                            json_start = line.find('{"timestamp"')
                        else:
                            json_start = line.find('{"timestamp"')
                        
                        if json_start != -1:
                            json_str = line[json_start:].strip()
                            entry = json.loads(json_str)
                            usage_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        usage_logger.error(f"Error parsing usage from app.log: {e}")
    
    return usage_entries

def rebuild_aggregate_from_app_log(log_file_path: str = "app.log"):
    """
    Rebuild the aggregate file from your existing app.log.
    
    Args:
        log_file_path: Path to your app.log file
    """
    usage_logger.info("Rebuilding aggregate from app.log...")
    
    # Parse all usage entries from app.log
    entries = parse_usage_from_app_log(log_file_path)
    
    if not entries:
        usage_logger.warning("No usage entries found in app.log")
        return
    
    # Reset aggregate
    aggregate = {
        "openai": {"total": 0, "by_module": {}, "by_function": {}},
        "whisper": {"total": 0, "by_module": {}, "by_function": {}}
    }
    
    # Process all entries
    for entry in entries:
        service = entry.get("service")
        module_name = entry.get("module", "unknown")
        function_name = entry.get("function", "")
        
        if service == "openai":
            # prompt_tokens = entry.get("prompt_tokens", 0)
            # completion_tokens = entry.get("completion_tokens", 0)
            usage = entry.get("total_tokens", 0)
        elif service == "whisper":
            usage = entry.get("duration_minutes", 0)
        else:
            continue
        
        # Update totals
        aggregate[service]["total"] += usage
        # aggregate[service]["prompt_tokens"] += prompt_tokens
        # aggregate[service]["completion_tokens"] += completion_tokens
        
        # Update by module
        if module_name not in aggregate[service]["by_module"]:
            aggregate[service]["by_module"][module_name] = 0
        aggregate[service]["by_module"][module_name] += usage
        
        # Update by function
        if function_name:
            function_key = f"{module_name}.{function_name}"
            if function_key not in aggregate[service]["by_function"]:
                aggregate[service]["by_function"][function_key] = 0
            aggregate[service]["by_function"][function_key] += usage
    
    # Save rebuilt aggregate
    try:
        with open(AGGREGATE_FILE, "w") as f:
            json.dump(aggregate, f, indent=2)
        usage_logger.info(f"Rebuilt aggregate with {len(entries)} entries")
        usage_logger.info(f"OpenAI total: {aggregate['openai']['total']} tokens")
        usage_logger.info(f"Whisper total: {aggregate['whisper']['total']:.2f} minutes")
    except Exception as e:
        usage_logger.error(f"Error saving rebuilt aggregate: {e}")

def reset_usage_logs():
    """Reset aggregate data (but keep app.log intact)."""
    if os.path.exists(AGGREGATE_FILE):
        os.remove(AGGREGATE_FILE)
        usage_logger.info("Token usage aggregate has been reset.")
    else:
        usage_logger.info("No aggregate file to reset.")
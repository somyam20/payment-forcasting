import os
import json
import time
import logging

# Import OpenAI configuration
from .openai_config import get_openai_client, get_chat_model_name, OPENAI_AVAILABLE, USE_AZURE

from .logger_config import setup_logger

setup_logger()

# Get the appropriate client and model name
client = get_openai_client()
MODEL = get_chat_model_name()

# Log connection status
if OPENAI_AVAILABLE:
    logging.info(f"OpenAI configured: {'Azure OpenAI' if USE_AZURE else 'Standard OpenAI'}")
    logging.info(f"Using model/deployment: {MODEL}")
else:
    logging.warning("OpenAI API not configured correctly")
    client = None
    OPENAI_AVAILABLE = False

def analyze_speech_for_screenshot_moments(speech_segments, teams_llm_config):
    """
    Analyze a list of speech segments using OpenAI to identify potential screenshot moments.
    
    Args:
        speech_segments: List of tuples (timestamp, speech_text)
        
    Returns:
        List of tuples (timestamp, reason) for potential screenshot moments
    """
    from .media_utils import analyze_speech_transcript
        
    if not speech_segments:
        return []
    
    # Combine speech segments into a single transcript with timestamps
    formatted_transcript = []
    for timestamp, text in speech_segments:
        formatted_transcript.append(f"[{timestamp:.2f}s] {text}")
    
    full_transcript = "\n".join(formatted_transcript)
    
    # Prompt for the LLM to analyze the transcript
    system_prompt = """You are an expert at analyzing product demo transcripts to identify 
    moments where screenshots should be captured for documentation.

    Task: Identify specific timestamps in the transcript where a screenshot would be 
    valuable for documentation purposes.

    Look for moments such as:
    1. When a new feature, screen, or UI element is being shown
    2. When an action is being performed (like clicking, opening menus, etc.)
    3. When important information or results are displayed
    4. Before and after significant state changes
    5. When something is highlighted or pointed out as important
    6. When new concepts or workflows are introduced
    7. When the speaker changes topics
    8. Whenever a visual change might be happening

    IMPORTANT: Be generous with identifying potential moments. Even if you're not 100% sure
    that a screenshot is needed, include the timestamp if it might be useful. It's better to
    suggest more timestamps than too few.

    If the transcript doesn't contain explicit UI references, still provide timestamps at
    approximately regular intervals throughout the transcript, especially at points where
    new topics or concepts are introduced.

    RESPONSE FORMAT: You MUST return your response as a valid JSON array of objects. 
    Each object must contain exactly these fields:
    - "timestamp": The timestamp in seconds (as a number, not string)
    - "reason": Brief explanation of why a screenshot is needed at this point
    - "importance": Rate from 1-5 where 5 is extremely important

    Example response (this is the EXACT format you must follow):
    [
        {"timestamp": 12.5, "reason": "Shows the main dashboard", "importance": 4},
        {"timestamp": 34.2, "reason": "Demonstrates the filter menu options", "importance": 3},
        {"timestamp": 50.1, "reason": "New topic introduced: reporting features", "importance": 3}
    ]

    Do not include any text before or after the JSON array. Return only valid JSON.
    """
    
    try:
        # Print the transcript being sent to OpenAI for debugging
        logging.info(f"\nSending transcript to OpenAI for analysis: {len(full_transcript)} characters")
        logging.info(f"First 200 chars: {full_transcript[:200]}...")
        
        # Use our utility function to analyze the transcript
        result = analyze_speech_transcript(
            transcript=f"Transcript:\n{full_transcript}", 
            prompt=system_prompt,
            teams_llm_config=teams_llm_config
        )
        
        # Convert the response to our internal format
        # Convert the response to our internal format
        screenshot_moments = []

        # Handle different possible response formats
        if isinstance(result, dict):
            if "timestamps" in result:
                # Format: {"timestamps": [...]}
                timestamps_data = result["timestamps"]
            elif "timestamp" in result:
                # Format: Single timestamp object (your current case)
                timestamps_data = [result]
            else:
                # Unknown dict format
                timestamps_data = []
        elif isinstance(result, list):
            # Format: Direct array of timestamp objects
            timestamps_data = result
        else:
            # Unexpected format
            timestamps_data = []

        # Process the timestamps
        for item in timestamps_data:
            if isinstance(item, dict) and "timestamp" in item and "reason" in item:
                timestamp = float(item["timestamp"])
                reason = f"AI detected: {item['reason']}"
                if "importance" in item:
                    reason += f" (Importance: {item['importance']}/5)"
                screenshot_moments.append((timestamp, reason))

        print(f"Found {len(screenshot_moments)} screenshot moments")
        for moment in screenshot_moments:
            print(f"  {moment[0]}s: {moment[1]}")        
        # If no screenshots were detected, create fallback timestamps at regular intervals
        if not screenshot_moments and speech_segments:
            print("\nNo screenshot moments detected by AI, creating fallback timestamps")
            duration = speech_segments[-1][0] - speech_segments[0][0]
            
            # If we have speech, use evenly distributed timestamps from the speech data
            num_fallback_shots = min(5, len(speech_segments))
            interval = max(1, len(speech_segments) // num_fallback_shots)
            
            for i in range(0, len(speech_segments), interval):
                if i < len(speech_segments):
                    timestamp = speech_segments[i][0]
                    text = speech_segments[i][1]
                    trimmed_text = text[:50] + "..." if len(text) > 50 else text
                    reason = f"AI fallback: Speech at timestamp ({trimmed_text})"
                    screenshot_moments.append((timestamp, reason))
        
        return screenshot_moments
        
    except Exception as e:
        logging.exception(f"Error analyzing speech with OpenAI: {e}")
        # Return an empty list as fallback
        return []

def analyze_transcript_chunk(transcript_chunk, teams_llm_config,min_importance=3):
    """
    Analyze a chunk of transcript using OpenAI to identify potential screenshot moments.
    
    Args:
        transcript_chunk: String containing transcript text
        min_importance: Minimum importance score (1-5) to include
        
    Returns:
        List of tuples (timestamp, reason) for potential screenshot moments
    """
    from .media_utils import analyze_speech_transcript
    
    system_prompt = """You are an expert at analyzing product demo transcripts to identify 
    moments where screenshots should be captured for documentation.

    Task: For each sentence, determine if it describes an action, UI element, feature, or 
    result that would be worth capturing in a screenshot.

    For each identified moment, assign an importance score from 1-5:
    5: Essential - A key feature or main screen that must be documented
    4: Important - Demonstrates a significant feature or interaction
    3: Useful - Shows useful information or minor features
    2: Minor - Might be useful in detailed documentation
    1: Optional - Only needed for comprehensive documentation

    Respond with a JSON object with a single key "analysis" containing an array, 
    where each item has:
    - "sentence": The relevant sentence or phrase
    - "screenshot_worthy": true or false
    - "importance": score from 1-5 
    - "reason": brief explanation why this deserves a screenshot
    
    Only mark sentences as screenshot_worthy if they clearly indicate 
    a visual element or action worth documenting.
    """
    
    try:
        # Use our utility function to analyze the transcript
        result = analyze_speech_transcript(
            transcript=transcript_chunk,
            prompt=system_prompt,
            teams_llm_config=teams_llm_config
        )
        
        screenshot_worthy_items = []
        if isinstance(result, dict) and "analysis" in result and isinstance(result["analysis"], list):
            for item in result["analysis"]:
                if item.get("screenshot_worthy", False) and item.get("importance", 0) >= min_importance:
                    screenshot_worthy_items.append({
                        "sentence": item.get("sentence", ""),
                        "importance": item.get("importance", 3),
                        "reason": item.get("reason", "Relevant visual element")
                    })
        
        return screenshot_worthy_items
        
    except Exception as e:
        logging.exception(f"Error analyzing transcript chunk with OpenAI: {e}")
        # Return an empty list as fallback
        return []
import cv2
import datetime
import os
from openai import OpenAI
from .openai_config import get_openai_client, USE_AZURE
import litellm
from litellm import completion
import whisper
from .api_usage_logger import log_openai_usage, log_whisper_usage
from src.utils.obs import LLMUsageTracker

token_tracker = LLMUsageTracker()

def transcribe_with_whisper(audio_file_path):
    """
    Transcribe audio using local Whisper model first, then fallback to OpenAI's Whisper API.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        
    Returns:
        str: Transcribed text, or empty string if transcription failed
    """
    # First try: Use local Whisper model if available
    try:
        from ..processors.audio.whisper_processor import get_optimized_whisper_processor
        
        whisper_processor = get_optimized_whisper_processor()
        if whisper_processor and whisper_processor.model is not None:
            print("Using local Whisper model for transcription")
            result = whisper_processor.transcribe_audio(audio_file_path)
            text = result.get('text', '').strip()
            if text:
                return text
            else:
                print("Local Whisper returned empty text, falling back to API")
    except Exception as e:
        print(f"Local Whisper model not available or failed: {e}, falling back to API")
    
    # Fallback: Use OpenAI Whisper API
    try:
        print("Using OpenAI Whisper API for transcription")
        std_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        
        # Get audio duration for logging
        audio_duration = _get_audio_duration(audio_file_path)
        file_size_mb = _get_file_size_mb(audio_file_path)
        
        with open(audio_file_path, "rb") as audio_file:
            response = std_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
        
        # Log Whisper usage
        log_whisper_usage(
            module_name=__name__,
            model="whisper-1",
            duration_seconds=audio_duration,
            function_name="transcribe_with_whisper",
            file_size_mb=file_size_mb
        )
        
        return response.text
    except Exception as e:
        print(f"Error transcribing with Whisper API: {e}")
        return ""

def analyze_speech_transcript(transcript, prompt, teams_llm_config):
    """
    Analyze a speech transcript using the configured OpenAI service.
    If Azure OpenAI is available, it will be used, otherwise fallback to standard OpenAI.
    
    Args:
        transcript (str): The transcribed text to analyze
        prompt (str): System prompt that tells the model what to do with the transcript
        
    Returns:
        dict: Analysis result as parsed from JSON response, or empty dict if failed
    """
    try:
        import json
        from typing import List, Dict, Any
        # from .openai_config import get_openai_client, get_chat_model_name, USE_AZURE
        from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
        
        # Get the appropriate client (Azure or standard OpenAI)
        # client = get_openai_client()
        # model = get_chat_model_name()
        
        # if client is None:
        #     print("OpenAI client not initialized. Cannot analyze speech transcript.")
        #     return {}
        
        # Prepare messages with proper typing
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system", 
            "content": prompt
        }
        
        user_message: ChatCompletionUserMessageParam = {
            "role": "user", 
            "content": transcript
        }
        
        print(f"Analyzing transcript with {'Azure OpenAI' if USE_AZURE else 'standard OpenAI'}")
        
        try:
            # if USE_AZURE:
            #     # For Azure OpenAI
            #     response = completion(
            #         model=f"azure/{model}",
            #         messages=[system_message, user_message],
            #         temperature=0.3,
            #         response_format={"type": "json_object"}
            #     )
                
            # else:
                # For standard OpenAI
            auth_token = teams_llm_config.pop("auth_token", "")
            response = completion(
                **teams_llm_config,
                messages=[system_message, user_message],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Extract usage information correctly
            # usage = response.usage
            if response:
                # log_openai_usage(
                #     module_name=__name__,
                #     # model=model,
                #     prompt_tokens=usage.prompt_tokens,
                #     completion_tokens=usage.completion_tokens,
                #     function_name="analyze_speech_transcript"
                # )
                token_tracker.track_response(response=response, auth_token=auth_token, model=teams_llm_config.get("model",""))
            
            # Parse and return the JSON response
            content = response.choices[0].message.content
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw content: {content}")
                    return {}
            
            return {}
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {}
    except Exception as e:
        print(f"Error in analyze_speech_transcript: {e}")
        return {}

def _get_audio_duration(audio_file_path):
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        float: Duration in seconds
    """
    try:
        import librosa
        # Use librosa if available
        duration = librosa.get_duration(path=audio_file_path)
        return duration
    except ImportError:
        try:
            # Fallback to using OpenCV for video files with audio
            cap = cv2.VideoCapture(audio_file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except:
            # If we can't determine duration, estimate based on file size
            # This is a rough estimate: ~1MB per minute for typical audio
            file_size_mb = _get_file_size_mb(audio_file_path)
            return file_size_mb * 60  # Rough estimate

def _get_file_size_mb(file_path):
    """
    Get file size in megabytes.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0

def get_video_info(video_path):
    """
    Extract information about a video file.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        dict: Dictionary containing video information.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    return info

def format_timestamp(seconds, for_filename=False):
    """
    Format time in seconds to a readable string.
    
    Args:
        seconds (float): Time in seconds.
        for_filename (bool): If True, format for use in a filename.
        
    Returns:
        str: Formatted time string.
    """
    if for_filename:
        # Format for filenames: 00h00m00s
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    else:
        # Format for display: 00:00:00
        time_obj = datetime.datetime.utcfromtimestamp(seconds)
        if seconds < 3600:  # Less than an hour
            return time_obj.strftime('%M:%S')
        else:
            return time_obj.strftime('%H:%M:%S')
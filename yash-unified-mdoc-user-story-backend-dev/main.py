#!/usr/bin/env python3
"""
Main application file for Meeting Document Generator
Contains all business logic functions that can be used without UI
"""

import sys
import io

# Fix Windows encoding issues FIRST
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
import tempfile
import time
import uuid
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime
from dotenv import load_dotenv
from litellm import completion
from src.utils.obs import LLMUsageTracker
import litellm

token_tracker = LLMUsageTracker()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

import os
import sys

# Set ffmpeg path BEFORE importing pydub
BIN_PATH = os.getenv('BIN_PATH', '')
# Set ffmpeg path BEFORE importing pydub
ffmpeg_bin = BIN_PATH if BIN_PATH else os.path.join(os.path.dirname(__file__), '..', 'ffmpeg-8.0-essentials_build', 'bin')

# Add to PATH
if ffmpeg_bin not in os.environ['PATH']:
    os.environ['PATH'] = ffmpeg_bin + os.pathsep + os.environ['PATH']

# Also explicitly set for pydub
os.environ['FFMPEG_BINARY'] = os.path.join(ffmpeg_bin, 'ffmpeg.exe')
os.environ['FFPROBE_BINARY'] = os.path.join(ffmpeg_bin, 'ffprobe.exe')

# NOW import pydub and other modules
from pydub import AudioSegment
from pydub.utils import which

# Import configuration
from src.utils.config_loader import get_config_loader
from src.utils.media_utils import get_video_info, format_timestamp
from src.utils.openai_config import get_openai_client, get_chat_model_name, OPENAI_AVAILABLE
from src.utils.audit_logger import Logger
from src.utils.cost_logger import UsageCostLogger
from src.utils.usage_cost_extractor import extract_token_usage_from_app_log
from src.utils.logger_config import setup_logger, setup_usage_logger

# Import processing components
from src.processors.video.video_processor import VideoProcessor
from src.processors.video.screenshot_extractor import ScreenshotExtractor
from src.document.document_generator import DocumentGenerator

# Setup logging
setup_logger()
usage_logger = setup_usage_logger()
audit_logger = Logger()
usage_log = UsageCostLogger()

# Load configuration
config_loader = get_config_loader()
app_config = config_loader.get_config('app_config.yaml')



def generate_session_id() -> str:
    """Generate a unique session ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    return f"session_{timestamp}_{unique_id}"


# def setup_ffmpeg_path():
#     """Setup FFmpeg path - try multiple locations"""
    
#     # List of possible FFmpeg locations to try
#     possible_paths = [
#         # Your actual installation location
#         os.path.join(os.path.dirname(__file__), '..', 'ffmpeg-8.0-essentials_build', 'bin'),
#         # Relative path from current directory
#         os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ffmpeg-8.0-essentials_build', 'bin')),
#         # Local ffmpeg folder (if you copy it later)
#         os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin'),
#         # Common Windows installations
#         r'C:\ffmpeg\bin',
#         r'C:\Program Files\ffmpeg\bin',
#     ]
    
#     for ffmpeg_bin_path in possible_paths:
#         ffmpeg_exe = os.path.join(ffmpeg_bin_path, 'ffmpeg.exe')
#         ffprobe_exe = os.path.join(ffmpeg_bin_path, 'ffprobe.exe')
        
#         if os.path.exists(ffmpeg_exe) and os.path.exists(ffprobe_exe):
#             # Add to PATH
#             os.environ['PATH'] = ffmpeg_bin_path + os.pathsep + os.environ['PATH']
#             print(f"✓ FFmpeg found and added to PATH: {ffmpeg_bin_path}")
#             return ffmpeg_bin_path
    
#     print("⚠ FFmpeg not found in expected locations:")
#     for path in possible_paths:
#         print(f"  - {path}")
#     return None


def get_video_duration_ffprobe(video_path: str) -> float:
    """
    Get video duration in minutes using ffprobe
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in minutes (rounded to 2 decimal places)
    """
    
    # Setup FFmpeg path early
    # ffmpeg_bin_path = setup_ffmpeg_path()
    
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration", "-of",
         "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return round(float(result.stdout) / 60, 2)

def extract_meeting_attendees(speech_segments: List[Tuple[float, str]], client: Any = None, model: str = None, teams_llm_config: Optional[Dict[str,Any]] = None) -> List[str]:
    """
    Extract unique speaker names/identities from speech segments using AI
    
    Args:
        speech_segments: List of (timestamp, speech_text) tuples
        client: OpenAI client (optional)
        model: Model name to use (optional)
    
    Returns:
        List of unique attendee names identified from the meeting
    """
    if not speech_segments or not OPENAI_AVAILABLE:
        return []
    
    try:
        # Combine all speech text
        full_transcript = "\n".join([text for _, text in speech_segments])
        
        # Limit transcript to avoid token limits
        if len(full_transcript) > 10000:
            full_transcript = full_transcript[:10000] + "..."
        
        # Create prompt to extract attendees
        prompt = f"""
        Analyze this meeting transcript and identify all unique speakers/attendees.
        
        TRANSCRIPT:
        {full_transcript}
        
        Your task:
        1. Identify each unique person speaking in the meeting
        2. Extract their names if mentioned (e.g., "Hi, I'm John" or "Thanks, Sarah")
        3. If names aren't explicitly mentioned, create generic identifiers (Speaker 1, Speaker 2, etc.)
        4. Return ONLY a JSON array of unique attendee names
        
        Response format:
        ["Name 1", "Name 2", "Name 3"]
        
        Do NOT include any other text, just the JSON array.
        """
        
        # if client is None:
        #     client = get_openai_client()
        # if model is None:
        #     model = get_chat_model_name()
        auth_token = teams_llm_config.pop("auth_token","")
        response = completion(
            **teams_llm_config,
            messages=[
                {"role": "system", "content": "You are an expert at identifying speakers in meetings. Extract attendee names from transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        token_tracker.track_response(response=response, auth_token=auth_token, model=teams_llm_config.get("model",""))
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        import re
        if response_text.startswith("```"):
            pattern = r'```(?:json)?\s*\n(.*?)\n```'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                formatted_content = match.group(1).strip()
            else:
                # If no match, fallback to original text or some safe default
                formatted_content = response_text.strip()
            
            attendees = json.loads(formatted_content)
        else:
            attendees = json.loads(response_text)
        
        logging.info(f"Extracted {len(attendees)} attendees from meeting: {attendees}")
        return attendees if isinstance(attendees, list) else []
        
    except Exception as e:
        logging.error(f"Error extracting attendees: {str(e)}")
        return []


def extract_meeting_highlights(speech_segments: List[Tuple[float, str]], client: Any = None, model: str = None, teams_llm_config: Optional[Dict[str,Any]] = None) -> List[str]:
    """
    Extract key discussion points and highlights from speech segments using AI
    
    Args:
        speech_segments: List of (timestamp, speech_text) tuples
        client: OpenAI client (optional)
        model: Model name to use (optional)
    
    Returns:
        List of key discussion points identified from the meeting
    """
    if not speech_segments or not teams_llm_config:
        return []
    
    try:
        # Combine all speech text
        full_transcript = "\n".join([text for _, text in speech_segments])
        
        # Limit transcript to avoid token limits
        if len(full_transcript) > 10000:
            full_transcript = full_transcript[:10000] + "..."
        
        # Create prompt to extract highlights
        prompt = f"""
        Analyze this meeting transcript and identify the key discussion points, decisions, and highlights.
        
        TRANSCRIPT:
        {full_transcript}
        
        Your task:
        1. Identify the most important discussion points
        2. Extract decisions made during the meeting
        3. Note timestamps where significant points were discussed
        4. Note action items and commitments
        5. Find key agreements and conclusions
        6. Return ONLY a JSON array of 5-8 key highlights
        
        Response format:
        ["Highlight 1: Description", "Highlight 2: Description"]
        
        Do NOT include any other text, just the JSON array.
        """
        
        # if client is None:
        #     client = get_openai_client()
        # if model is None:
        #     model = get_chat_model_name()
        auth_token = teams_llm_config.pop("auth_token","")
        response = completion(
            **teams_llm_config,
            messages=[
                {"role": "system", "content": "You are an expert at identifying key points in meetings. Extract the most important discussion points from transcripts."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        
        token_tracker.track_response(response=response, auth_token=auth_token, model=teams_llm_config.get("model",""))
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        import re
        if response_text.startswith("```"):
            pattern = r'```(?:json)?\s*\n(.*?)\n```'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                formatted_content = match.group(1).strip()
            else:
                # If no match, fallback to original text or some safe default
                formatted_content = response_text.strip()
            
            highlights = json.loads(formatted_content)
        else:
            highlights = json.loads(response_text)
        
        logging.info(f"Extracted {len(highlights)} highlights from meeting")
        return highlights if isinstance(highlights, list) else []
        
    except Exception as e:
        logging.error(f"Error extracting highlights: {str(e)}")
        return []

def process_video(
    video_path: str,
    client_name: str,
    detection_mode: str = "basic",
    use_speech: bool = True,
    use_mouse_detection: bool = True,
    use_scene_detection: bool = False,
    use_ai_analysis: bool = True,
    capture_after_interaction: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    session_guid: Optional[str] = None,
    teams_llm_config: Optional[Dict[str,Any]] = None
) -> Dict[str, Any]:
    """
    Process a video file to extract screenshots and speech data
    
    Args:
        video_path: Path to the video file
        client_name: Name of the client
        detection_mode: "basic" or "advanced"
        use_speech: Enable speech-based keyword detection
        use_mouse_detection: Enable mouse cursor tracking
        use_scene_detection: Enable scene change detection
        use_ai_analysis: Enable AI-powered content analysis
        capture_after_interaction: Capture screenshots after interactions
        progress_callback: Optional callback function(progress_pct, message)
        session_guid: Optional session GUID for logging
        
    Returns:
        Dictionary containing:
        - screenshots: List of (image, timestamp, reason) tuples
        - speech_timestamps: List of (timestamp, text) tuples
        - keyword_results: List of keyword detection results
        - processing_time: Time taken in seconds
        - extractor: ScreenshotExtractor instance
    """
    if session_guid is None:
        session_guid = str(uuid.uuid4())
    
    session_id = generate_session_id()
    usage_logger.info(f"--- SESSION START: {session_id} ---")
    
    start_time = time.time()
    
    try:
        # Initialize video processor
        processor = VideoProcessor(video_path)
        
        # Configure screenshot extractor
        threshold = 25
        min_area = 2000
        text_change_threshold = 10
        cooldown = 3
        ssim_threshold = 0.85
        
        extractor = ScreenshotExtractor(
            threshold=threshold,
            min_area=min_area,
            text_change_threshold=text_change_threshold,
            cooldown=cooldown,
            ssim_threshold=ssim_threshold,
            keyword_trigger=use_speech,
            mouse_tracking=use_mouse_detection,
            use_ai_speech_analysis=bool(use_ai_analysis and OPENAI_AVAILABLE),
            detection_mode=detection_mode,
        )
        
        fps = processor.fps
        frame_count = processor.frame_count
        
        # Extract audio if needed
        if use_speech or use_ai_analysis:
            if progress_callback:
                progress_callback(0.05, "Extracting audio...")
            success = extractor.extract_audio_from_video(video_path)
            if not success and not OPENAI_AVAILABLE:
                logging.warning("Audio extraction failed and OpenAI not available")
        
        # Process video frames
        if progress_callback:
            progress_callback(0.1, f"Processing {frame_count} frames...")
        
        screenshots = extractor.two_phase_process(
            processor,
            fps,
            frame_count,
            progress_callback=progress_callback,
            teams_llm_config=teams_llm_config
        )
        
        # Deduplication logic
        grouped_screenshots = {}
        for img, ts, reason in screenshots:
            rounded_ts = round(ts * 2) / 2
            if rounded_ts not in grouped_screenshots:
                grouped_screenshots[rounded_ts] = []
            grouped_screenshots[rounded_ts].append((img, ts, reason))
        
        deduplicated_screenshots = []
        for ts, group in grouped_screenshots.items():
            if len(group) == 1:
                deduplicated_screenshots.append(group[0])
            else:
                # Prioritize keyword triggers, then AI detected, then others
                keyword_screenshots = [s for s in group if "Keyword trigger" in s[2]]
                if keyword_screenshots:
                    deduplicated_screenshots.append(keyword_screenshots[0])
                    continue
                
                ai_screenshots = [s for s in group if "AI detected" in s[2]]
                if ai_screenshots:
                    deduplicated_screenshots.append(ai_screenshots[0])
                    continue
                
                deduplicated_screenshots.append(group[0])
        
        screenshots = deduplicated_screenshots
        processing_time = time.time() - start_time
        
        # Extract speech timestamps
        speech_timestamps = []
        if hasattr(extractor, 'speech_timestamps') and extractor.speech_timestamps:
            speech_timestamps = extractor.speech_timestamps
            logging.info(f" Loaded {len(speech_timestamps)} speech segments")
        
        # Extract keyword results
        keyword_results = []
        if hasattr(extractor, 'keyword_results'):
            keyword_results = extractor.keyword_results
        
        # Log usage costs
        video_filename = os.path.basename(video_path)
        video_size = os.path.getsize(video_path)
        video_size_mb = round(video_size / (1024 * 1024), 2)
        video_duration = get_video_duration_ffprobe(video_path)
        
        whisper_cost = float(os.getenv("AZURE_WHISPER_CLIENT_COST", "0"))
        token_usage_cost = extract_token_usage_from_app_log(session_id=session_guid)
        token_usage_cost["whisper_cost"] = round(video_duration * whisper_cost, 2)
        
        usage_log.log(
            guid=session_guid,
            client_name=client_name,
            tool_name="mDoc_v2",
            file_name=video_filename.split(".")[0],
            file_type=video_filename.split(".")[-1],
            file_size=f"{video_size_mb} MB",
            video_duration=video_duration,
            screenshot_processing=round(token_usage_cost["openai_cost"] + token_usage_cost["whisper_cost"], 2),
            document_generation=0,
            document_type="",
            prompt_tokens=token_usage_cost["prompt_tokens"],
            completion_tokens=token_usage_cost["completion_tokens"],
            total_tokens=token_usage_cost["total_tokens"],
            open_ai_cost=token_usage_cost["openai_cost"],
            azure_whisper_cost=token_usage_cost["whisper_cost"],
            total_cost=round(token_usage_cost["openai_cost"] + token_usage_cost["whisper_cost"], 2)
        )

        meeting_attendees = []
        meeting_highlights = []
        
        if use_speech and speech_timestamps:
            meeting_attendees = extract_meeting_attendees(speech_timestamps, teams_llm_config=teams_llm_config)
            meeting_highlights = extract_meeting_highlights(speech_timestamps, teams_llm_config=teams_llm_config)
        
        return {
            "screenshots": screenshots,
            "speech_timestamps": speech_timestamps,
            "keyword_results": keyword_results,
            "processing_time": processing_time,
            "extractor": extractor,
            "session_id": session_id,
            "session_guid": session_guid,
            "meeting_attendees": meeting_attendees,
            "meeting_highlights": meeting_highlights  
        }
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}", exc_info=True)
        raise


def generate_document(
    video_path: str,
    screenshots: List[Tuple[Any, float, str]],
    client_name: str,
    doc_title: str,
    doc_type: str = "user_story_generator",
    doc_format: str = "PDF",
    speech_segments: Optional[List[Tuple[float, str]]] = None,
    enable_missing_questions: bool = True,
    enable_process_map: bool = True,
    include_screenshots: bool = True,
    session_guid: Optional[str] = None,
    meeting_participants: Optional[List[str]] = None,
    meeting_highlights: Optional[List[str]] = None,
    teams_llm_config: Optional[Dict[str,Any]] = None
) -> Dict[str, Any]:
    """
    Generate a document from processed video data
    
    Args:
        video_path: Path to the video file
        screenshots: List of (image, timestamp, reason) tuples
        client_name: Name of the client
        doc_title: Title for the document
        doc_type: Type of document ("kt_document", "meeting_summary", "user_story_generator", "general_documentation")
        doc_format: Output format ("PDF", "DOCX", "Both")
        speech_segments: Optional list of (timestamp, text) tuples
        enable_missing_questions: Include missing questions section
        enable_process_map: Include process map diagram
        include_screenshots: Include screenshots in document
        session_guid: Optional session GUID for logging
        
    Returns:
        Dictionary containing:
        - pdf_bytes: PDF document bytes (if PDF or Both)
        - docx_bytes: DOCX document bytes (if DOCX or Both)
        - title: Document title
        - format: Document format
    """
    if session_guid is None:
        session_guid = str(uuid.uuid4())
    
    session_id = generate_session_id()
    usage_logger.info(f"--- SESSION START: {session_id} ---")
    
    try:

        video_duration = get_video_duration_ffprobe(video_path)
        # Set description based on document type
        doc_type_descriptions = {
            "user_story_generator": "Collection of user stories with acceptance criteria from requirements discussions."
        }
        doc_description = doc_type_descriptions.get(doc_type, doc_type_descriptions["user_story_generator"])
        
        # Prepare speech segments
        if speech_segments is None:
            speech_segments = []
        
        # Create document generator
        doc_generator = DocumentGenerator(
            video_path,
            screenshots,
            use_ai=True,
            title=doc_title,
            description=doc_description,
            speech_segments=speech_segments,
            document_type=doc_type,
            generate_missing_questions=enable_missing_questions,
            generate_process_map=enable_process_map,
            include_screenshots=include_screenshots,
            meeting_participants=meeting_participants or [],
            meeting_highlights=meeting_highlights or [],
            meeting_duration_minutes= video_duration,
            session_guid=session_guid,
            teams_llm_config=teams_llm_config
        )
        
        # Generate documents based on format selection
        pdf_bytes = None
        docx_bytes = None
        
        if doc_format in ["Both", "PDF"]:
            pdf_bytes = doc_generator.get_document_bytes("pdf")
        
        if doc_format in ["Both", "WORD", "DOCX"]:
            docx_bytes = doc_generator.get_document_bytes("docx")
        
        # Log usage costs
        video_filename = os.path.basename(video_path)
        video_size = os.path.getsize(video_path)
        video_size_mb = round(video_size / (1024 * 1024), 2)
        # video_duration = get_video_duration_ffprobe(video_path)
        
        token_usage_cost = extract_token_usage_from_app_log(session_id=session_guid)
        token_usage_cost["whisper_cost"] = 0
        
        usage_log.log(
            guid=session_guid,
            client_name=client_name,
            tool_name="mDoc_v2",
            file_name=video_filename.split(".")[0],
            file_type=video_filename.split(".")[-1],
            file_size=f"{video_size_mb} MB",
            video_duration=video_duration,
            document_type=doc_type,
            screenshot_processing=0,
            document_generation=round(token_usage_cost["openai_cost"] + token_usage_cost["whisper_cost"], 2),
            prompt_tokens=token_usage_cost["prompt_tokens"],
            completion_tokens=token_usage_cost["completion_tokens"],
            total_tokens=token_usage_cost["total_tokens"],
            open_ai_cost=token_usage_cost["openai_cost"],
            azure_whisper_cost=token_usage_cost["whisper_cost"],
            total_cost=round(token_usage_cost["openai_cost"] + token_usage_cost["whisper_cost"], 2)
        )
        
        # Log audit
        audit_logger.log(
            client_name=client_name,
            tool_name="mDoc_v2",
            file_name=video_filename.split(".")[0],
            file_type=video_filename.split(".")[-1],
            file_size=f"{video_size_mb} MB",
        )
        
        return {
            "pdf_bytes": pdf_bytes,
            "docx_bytes": docx_bytes,
            "title": doc_title,
            "format": doc_format,
            "doc_type": doc_type
        }
        
    except Exception as e:
        logging.error(f"Error generating document: {str(e)}", exc_info=True)
        raise


def process_video_and_generate_document(
    video_path: str,
    client_name: str,
    doc_title: str,
    doc_type: str = "user_story_generator",
    doc_format: str = "PDF",
    detection_mode: str = "basic",
    use_speech: bool = True,
    use_mouse_detection: bool = True,
    use_scene_detection: bool = False,
    use_ai_analysis: bool = True,
    enable_missing_questions: bool = True,
    enable_process_map: bool = True,
    include_screenshots: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    meeting_participants: Optional[List[str]] = None,
    meeting_highlights: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Complete pipeline: Process video and generate document
    
    Args:
        video_path: Path to the video file
        client_name: Name of the client
        doc_title: Title for the document
        doc_type: Type of document
        doc_format: Output format ("PDF", "DOCX", "Both")
        detection_mode: "basic" or "advanced"
        use_speech: Enable speech-based keyword detection
        use_mouse_detection: Enable mouse cursor tracking
        use_scene_detection: Enable scene change detection
        use_ai_analysis: Enable AI-powered content analysis
        enable_missing_questions: Include missing questions section
        enable_process_map: Include process map diagram
        include_screenshots: Include screenshots in document
        progress_callback: Optional callback function(progress_pct, message)
        
    Returns:
        Dictionary containing processing results and generated documents
    """
    session_guid = str(uuid.uuid4())
    
    # Step 1: Process video
    if progress_callback:
        progress_callback(0.0, "Starting video processing...")
    
    processing_result = process_video(
        video_path=video_path,
        client_name=client_name,
        detection_mode=detection_mode,
        use_speech=use_speech,
        use_mouse_detection=use_mouse_detection,
        use_scene_detection=use_scene_detection,
        use_ai_analysis=use_ai_analysis,
        progress_callback=progress_callback,
        session_guid=session_guid
    )
    
    # Step 2: Generate document
    if progress_callback:
        progress_callback(0.9, "Generating document...")
    
    document_result = generate_document(
        video_path=video_path,
        screenshots=processing_result["screenshots"],
        client_name=client_name,
        doc_title=doc_title,
        doc_type=doc_type,
        doc_format=doc_format,
        speech_segments=processing_result["speech_timestamps"],
        enable_missing_questions=enable_missing_questions,
        enable_process_map=enable_process_map,
        include_screenshots=include_screenshots,
        session_guid=session_guid,
        meeting_participants=meeting_participants,
        meeting_highlights=meeting_highlights
    )
    
    if progress_callback:
        progress_callback(1.0, "Complete!")
    
    return {
        **processing_result,
        **document_result
    }


def main_cli():
    """
    CLI entry point for running the application without UI
    Example usage:
        python main.py --video path/to/video.mp4 --client "Client Name" --title "Document Title"
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Meeting Document Generator - CLI Mode")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--client", required=True, help="Client name")
    parser.add_argument("--title", required=True, help="Document title")
    parser.add_argument("--doc-type", default="user_story_generator", 
                       choices=["user_story_generator"],
                       help="Document type")
    parser.add_argument("--format", default="PDF", choices=["PDF", "DOCX", "Both"],
                       help="Output format")
    parser.add_argument("--mode", default="basic", choices=["basic", "advanced"],
                       help="Processing mode")
    parser.add_argument("--output-dir", default=".", help="Output directory for documents")
    parser.add_argument("--participants", nargs="+", default=[], 
                       help="Meeting participants (e.g., --participants 'John Doe' 'Jane Smith')")
    parser.add_argument("--highlights", nargs="+", default=[], 
                       help="Key discussion points (e.g., --highlights 'Budget approved' 'Timeline discussed')")
    parser.add_argument("--no-missing-questions", action="store_true",
                       help="Disable missing questions section")
    parser.add_argument("--no-process-map", action="store_true",
                       help="Disable process map diagram")
    parser.add_argument("--no-screenshots", action="store_true",
                       help="Exclude screenshots from document")
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Progress callback for CLI
    def progress_callback(progress, message):
        print(f"[{int(progress*100)}%] {message}")
    
    try:
        print(f"Processing video: {args.video}")
        print(f"Client: {args.client}")
        print(f"Document Title: {args.title}")
        print(f"Document Type: {args.doc_type}")
        print(f"Format: {args.format}")
        print(f"Mode: {args.mode}")
        print("-" * 50)
        
        result = process_video_and_generate_document(
            video_path=args.video,
            client_name=args.client,
            doc_title=args.title,
            doc_type=args.doc_type,
            doc_format=args.format,
            detection_mode=args.mode,
            enable_missing_questions=not args.no_missing_questions,
            enable_process_map=not args.no_process_map,
            include_screenshots=not args.no_screenshots,
            progress_callback=progress_callback,
            meeting_participants=args.participants,
            meeting_highlights=args.highlights
        )
        
        # Save documents
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        today_date = datetime.now().strftime("%Y-%m-%d")
        doc_title_safe = args.title.replace(" ", "_")
        
        if result.get("pdf_bytes"):
            pdf_path = output_dir / f"{doc_title_safe}_{args.doc_type}_{today_date}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(result["pdf_bytes"])
            print(f" PDF saved: {pdf_path}")
        
        if result.get("docx_bytes"):
            docx_path = output_dir / f"{doc_title_safe}_{args.doc_type}_{today_date}.docx"
            with open(docx_path, "wb") as f:
                f.write(result["docx_bytes"])
            print(f" DOCX saved: {docx_path}")
        
        print(f"\n Processing complete!")
        print(f"   Screenshots extracted: {len(result['screenshots'])}")
        print(f"   Speech segments: {len(result['speech_timestamps'])}")
        print(f"   Processing time: {result['processing_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logging.exception("Error in CLI mode")
        sys.exit(1)


if __name__ == "__main__":
    # Add ffmpeg to PATH
    BIN_PATH = os.getenv('BIN_PATH', '')
    ffmpeg_path = BIN_PATH if BIN_PATH else os.path.join(os.path.dirname(__file__), '..', 'ffmpeg-8.0-essentials_build', 'bin')
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    main_cli()


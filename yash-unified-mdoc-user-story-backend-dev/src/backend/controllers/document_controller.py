"""
Document controller for handling meeting document processing
"""

import os
import uuid
import logging
import zipfile
import io
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from urllib.parse import unquote, urlparse
from fastapi import UploadFile, HTTPException

# Import business logic from main.py
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from main import (
    process_video,
    generate_document,
    get_video_duration_ffprobe
)
from src.utils.media_utils import get_video_info
from src.utils.s3_utility import S3Utility

logger = logging.getLogger(__name__)

# In-memory storage for processed data (screenshots, etc.)
# In production, consider using Redis or a database
_session_storage: Dict[str, Dict[str, Any]] = {}
s3_utility = S3Utility()
TEMP_DATA_DIR = Path("data/temp")
DOCUMENTS_S3_FOLDER = os.getenv("S3_DOCUMENT_FOLDER", "meeting-documents")


def _ensure_temp_dir() -> None:
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_bytes_to_temp_file(file_content: bytes, file_suffix: str) -> str:
    _ensure_temp_dir()
    suffix = file_suffix if file_suffix else ".tmp"
    temp_filename = f"{uuid.uuid4()}{suffix}"
    temp_path = TEMP_DATA_DIR / temp_filename
    with open(temp_path, "wb") as temp_file:
        temp_file.write(file_content)
    return str(temp_path)


def _extract_filename_and_suffix_from_url(file_url: str) -> Tuple[str, str]:
    parsed_url = urlparse(unquote(file_url))
    filename = Path(parsed_url.path).name or f"video_{uuid.uuid4()}.mp4"
    return filename, Path(filename).suffix.lower()


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Path to saved file
    """
    # Create temp directory if it doesn't exist
    _ensure_temp_dir()
    
    # Generate unique filename
    file_ext = Path(upload_file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_ext}"
    temp_path = TEMP_DATA_DIR / temp_filename
    
    try:
        # Save file
        with open(temp_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)
        
        logger.info(f"Saved uploaded file to {temp_path}")
        return str(temp_path)
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


async def process_meeting(
    file: Optional[UploadFile],
    file_url: Optional[str],
    client_name: str,
    detection_mode: str = "basic",
    use_speech: bool = True,
    use_mouse_detection: bool = True,
    use_scene_detection: bool = False,
    use_ai_analysis: bool = True,
    team_llm_config: Optional[Dict[str,Any]] = None
) -> Dict[str, Any]:
    """
    Process uploaded meeting video file
    
    Args:
        file: Uploaded video file
        file_url: S3 URL to the video file
        client_name: Name of the client
        detection_mode: "basic" or "advanced"
        use_speech: Enable speech-based keyword detection
        use_mouse_detection: Enable mouse cursor tracking
        use_scene_detection: Enable scene change detection
        use_ai_analysis: Enable AI-powered content analysis
        
    Returns:
        Dictionary containing processing results
    """
    if not file and not file_url:
        raise HTTPException(
            status_code=400,
            detail="Either a file upload or an S3 file URL must be provided."
        )

    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_path: Optional[str] = None
    original_filename: str = ""
    file_ext: str = ""

    if file:
        file_ext = Path(file.filename).suffix.lower()
        original_filename = file.filename
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(sorted(allowed_extensions))}"
            )
        video_path = await save_uploaded_file(file)
    else:
        original_filename, file_ext = _extract_filename_and_suffix_from_url(file_url)
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type from S3 URL. Allowed: {', '.join(sorted(allowed_extensions))}"
            )
        try:
            file_bytes = s3_utility.get_data_from_s3_by_url(file_url)
        except HTTPException:
            raise
        video_path = _save_bytes_to_temp_file(file_bytes, file_ext or ".mp4")
    
    session_guid = str(uuid.uuid4())
    
    try:
        # Process video
        result = process_video(
            video_path=video_path,
            client_name=client_name,
            detection_mode=detection_mode,
            use_speech=use_speech,
            use_mouse_detection=use_mouse_detection,
            use_scene_detection=use_scene_detection,
            use_ai_analysis=use_ai_analysis,
            progress_callback=None,  # No progress callback for API
            session_guid=session_guid,
            teams_llm_config=team_llm_config
        )
        
        # Get video info
        video_info = get_video_info(video_path)
        video_duration = get_video_duration_ffprobe(video_path)
        
        # Store screenshots and other data in session storage
        # Convert screenshots to serializable format (store metadata)
        screenshots_metadata = []
        for img, timestamp, reason in result["screenshots"]:
            screenshots_metadata.append({
                "timestamp": timestamp,
                "reason": reason
            })
        
        # Store full data in memory for later use
        _session_storage[session_guid] = {
            "screenshots": result["screenshots"],  # Full screenshot objects
            "speech_timestamps": result["speech_timestamps"],
            "keyword_results": result.get("keyword_results", []),
            "video_path": video_path,
            "client_name": client_name
        }
        
        # Convert transcript to list of dicts for JSON serialization
        transcript = []
        for timestamp, text in result["speech_timestamps"]:
            transcript.append({
                "timestamp": timestamp,
                "text": text
            })
        
        # Prepare response with full transcript (like Streamlit)
        response = {
            "success": True,
            "session_guid": result["session_guid"],
            "session_id": result["session_id"],
            "video_path": video_path,
            "video_info": {
                "filename": os.path.basename(original_filename or video_path),
                "duration_minutes": video_duration,
                "fps": video_info["fps"],
                "frame_count": video_info["frame_count"],
                "width": video_info["width"],
                "height": video_info["height"]
            },
            "screenshots": screenshots_metadata,  # Metadata only
            "screenshots_count": len(result["screenshots"]),
            "transcript": transcript,  # Full transcript like Streamlit
            "speech_segments": transcript,  # Alias for compatibility
            "speech_segments_count": len(result["speech_timestamps"]),
            "keyword_results": result.get("keyword_results", []),
            "processing_time": result["processing_time"],
            "message": "Video processed successfully"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing meeting: {e}", exc_info=True)
        # Clean up temp file on error
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


async def generate_meeting_document(
    doc_title: str,
    doc_type: str = "user_story_generator",
    doc_format: str = "PDF",
    enable_missing_questions: bool = True,
    enable_process_map: bool = True,
    include_screenshots: bool = True,
    session_guid: Optional[str] = None,
    video_path: Optional[str] = None,
    client_name: Optional[str] = None,
    teams_llm_config: Optional[Dict[str,Any]] = None
) -> Dict[str, Any]:
    """
    Generate document from processed video
    
    Args:
        doc_title: Title for the document
        doc_type: Type of document
        doc_format: Output format ("PDF", "DOCX", "Both")
        enable_missing_questions: Include missing questions section
        enable_process_map: Include process map diagram
        include_screenshots: Include screenshots in document
        session_guid: Session GUID from upload endpoint (required if video_path/client_name not provided)
        video_path: Path to video file (optional if session_guid provided)
        client_name: Name of the client (optional if session_guid provided)
        
    Returns:
        Dictionary containing generated document info
    """
    try:
        # If session_guid is provided, retrieve all data from storage
        if session_guid:
            if session_guid not in _session_storage:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_guid} not found. Please upload and process video first."
                )
            
            stored_data = _session_storage[session_guid]
            video_path = stored_data.get("video_path")
            client_name = stored_data.get("client_name")
            screenshots = stored_data.get("screenshots")
            speech_segments = stored_data.get("speech_timestamps")
            
            if not video_path:
                raise HTTPException(
                    status_code=400,
                    detail="Session data incomplete. Missing video_path."
                )
            if not client_name:
                raise HTTPException(
                    status_code=400,
                    detail="Session data incomplete. Missing client_name."
                )
            
            logger.info(f"Retrieved all data from session storage for {session_guid}")
        else:
            # If no session_guid, video_path and client_name are required
            if not video_path or not client_name:
                raise HTTPException(
                    status_code=400,
                    detail="Either session_guid must be provided, or both video_path and client_name are required."
                )
            
            # Check if video file exists
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video file not found")
            
            # If screenshots and speech_segments are not provided, process video first
            screenshots = None
            speech_segments = None
            if screenshots is None or speech_segments is None:
                logger.info("Processing video to get screenshots and transcript")
                session_guid = str(uuid.uuid4())
                # Process video to get screenshots and transcript
                process_result = process_video(
                    video_path=video_path,
                    client_name=client_name,
                    detection_mode="basic",
                    session_guid=session_guid
                )
                screenshots = process_result["screenshots"]
                speech_segments = process_result["speech_timestamps"]
        
        # Convert transcript from API format (list of dicts) to expected format (list of tuples)
        if speech_segments and isinstance(speech_segments, list) and len(speech_segments) > 0:
            if isinstance(speech_segments[0], dict):
                # Convert from API format: [{"timestamp": 1.0, "text": "..."}]
                # To expected format: [(1.0, "...")]
                speech_segments = [(seg["timestamp"], seg["text"]) for seg in speech_segments]
        
        # Generate document
        result = generate_document(
            video_path=video_path,
            screenshots=screenshots,
            client_name=client_name,
            doc_title=doc_title,
            doc_type=doc_type,
            doc_format=doc_format,
            speech_segments=speech_segments,
            enable_missing_questions=enable_missing_questions,
            enable_process_map=enable_process_map,
            include_screenshots=include_screenshots,
            session_guid=session_guid,
            teams_llm_config=teams_llm_config
        )
        
        # Return file(s) directly as downloadable
        doc_title_safe = result["title"].replace(" ", "_")
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        session_folder = session_guid or str(uuid.uuid4())
        s3_folder = f"{DOCUMENTS_S3_FOLDER}/{session_folder}".strip("/")
        base_filename = f"{doc_title_safe}_{result['doc_type']}_{today_date}"

        def _upload_and_presign(file_bytes: bytes, filename: str) -> Tuple[str, str]:
            file_url = s3_utility.upload_file(file_bytes, filename, s3_folder)
            presigned_url = s3_utility.generate_presigned_url(file_url)
            return file_url, presigned_url

        upload_details: Dict[str, Any] = {}

        if result["format"] == "Both":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if result.get("pdf_bytes"):
                    zip_file.writestr(f"{base_filename}.pdf", result["pdf_bytes"])
                if result.get("docx_bytes"):
                    zip_file.writestr(f"{base_filename}.docx", result["docx_bytes"])
            zip_buffer.seek(0)
            if zip_buffer.getbuffer().nbytes == 0:
                return {
                    "success": False,
                    "message": "No documents generated for upload.",
                    "doc_format": result["format"]
                }
            s3_url, presigned_url = _upload_and_presign(zip_buffer.getvalue(), f"{base_filename}.zip")
            upload_details = {
                "s3_url": s3_url,
                "download_url": presigned_url,
                "file_name": f"{base_filename}.zip",
                "content_type": "application/zip"
            }

        elif result["format"] == "PDF" and result.get("pdf_bytes"):
            s3_url, presigned_url = _upload_and_presign(result["pdf_bytes"], f"{base_filename}.pdf")
            upload_details = {
                "s3_url": s3_url,
                "download_url": presigned_url,
                "file_name": f"{base_filename}.pdf",
                "content_type": "application/pdf"
            }

        elif result["format"] in ["DOCX", "WORD"] and result.get("docx_bytes"):
            s3_url, presigned_url = _upload_and_presign(result["docx_bytes"], f"{base_filename}.docx")
            upload_details = {
                "s3_url": s3_url,
                "download_url": presigned_url,
                "file_name": f"{base_filename}.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }

        else:
            return {
                "success": False,
                "message": "No document generated. Please check your parameters.",
                "has_pdf": result.get("pdf_bytes") is not None,
                "has_docx": result.get("docx_bytes") is not None
            }

        return {
            "success": True,
            "session_guid": session_guid,
            "doc_title": result["title"],
            "doc_type": result["doc_type"],
            "doc_format": result["format"],
            **upload_details
        }
        
    except Exception as e:
        logger.error(f"Error generating document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")


"""
Document API routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query, Request
from typing import Optional
from src.backend.controllers import document_controller
from src.utils.config import get_model_config
import logging
import json

router = APIRouter()

@router.post("/upload")
async def upload_meeting(
    req: Request,
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    client_name: str = Form(...),
    user_metadata: str = Form(...),
    detection_mode: str = Form("basic"),
    use_speech: bool = Form(True),
    use_mouse_detection: bool = Form(True),
    use_scene_detection: bool = Form(False),
    use_ai_analysis: bool = Form(True)
):
    """
    Upload and process meeting recording
    
    - **file_url**: HTTPS/S3 URL pointing to a video already in S3 (preferred)
    - **file**: Legacy multipart video upload (MP4, AVI, MOV, MKV)
    - **client_name**: Name of the client
    - **detection_mode**: Processing mode ("basic" or "advanced")
    - **use_speech**: Enable speech-based keyword detection
    - **use_mouse_detection**: Enable mouse cursor tracking
    - **use_scene_detection**: Enable scene change detection
    - **use_ai_analysis**: Enable AI-powered content analysis
    
    Returns:
    - **transcript**: Full transcript with timestamps (list of {timestamp, text})
    - **screenshots**: Screenshot metadata (list of {timestamp, reason})
    - **session_guid**: Use this for document generation / download
    - **video_path**: Temporary path used during processing
    - Processing statistics and video info
    """
    if not file and not file_url:
        raise HTTPException(status_code=400, detail="Provide either file upload or file_url.")
    
    user_metadata = json.loads(user_metadata) if user_metadata else {}
    team_id = user_metadata.get("team_id")
    
    try:
        async with get_model_config() as config:
                # Get the team's model configuration
                team_config = await config.get_team_model_config(team_id)
                model = team_config["selected_model"]
                provider = team_config["provider"]
                provider_model = f"{provider}/{model}"
                model_config = team_config["config"]
            
                # Create LLM instance with the team's configuration
                llm_params = {
                    "model": provider_model,
                    **model_config  
                }

                llm_params["auth_token"] = req.headers.get("Authorization","")
           
    except Exception as e:
        logging.error(f"Failed to create LLM instance for team {team_id}: {str(e)}")
        raise ValueError(f"Failed to get model configuration for team {team_id}: {str(e)}")

    return await document_controller.process_meeting(
        file=file,
        file_url=file_url,
        client_name=client_name,
        detection_mode=detection_mode,
        use_speech=use_speech,
        use_mouse_detection=use_mouse_detection,
        use_scene_detection=use_scene_detection,
        use_ai_analysis=use_ai_analysis,
        team_llm_config=llm_params
    )


@router.post("/generate/user-story-generator")
async def generate_user_story_generator(
    req: Request,
    doc_title: str = Form(...),
    session_guid: str = Form(...),
    user_metadata: str = Form(...),
    doc_format: str = Form("PDF"),
    enable_missing_questions: bool = Form(True),
    enable_process_map: bool = Form(True),
    include_screenshots: bool = Form(True),
    video_path: Optional[str] = Form(None),
    client_name: Optional[str] = Form(None),
):
    """
    Generate Meeting Summary Document from processed video
    
    - **session_guid**: Session GUID from upload endpoint (REQUIRED)
    - **doc_title**: Title for the document
    - **doc_format**: Output format ("PDF", "DOCX", "Both")
    - **enable_missing_questions**: Include missing questions section
    - **enable_process_map**: Include process map diagram
    - **include_screenshots**: Include screenshots in document
    - **video_path**: Optional - only needed if session_guid not provided
    - **client_name**: Optional - only needed if session_guid not provided
    
    Returns:
    - JSON response containing an S3 presigned download URL for the generated file
    """

    user_metadata = json.loads(user_metadata) if user_metadata else {}
    team_id = user_metadata.get("team_id")

    try:
        async with get_model_config() as config:
                # Get the team's model configuration
                team_config = await config.get_team_model_config(team_id)
                model = team_config["selected_model"]
                provider = team_config["provider"]
                provider_model = f"{provider}/{model}"
                model_config = team_config["config"]
            
                # Create LLM instance with the team's configuration
                llm_params = {
                    "model": provider_model,
                    **model_config  
                }

                llm_params["auth_token"] = req.headers.get("Authorization","")
           
    except Exception as e:
        logging.error(f"Failed to create LLM instance for team {team_id}: {str(e)}")
        raise ValueError(f"Failed to get model configuration for team {team_id}: {str(e)}")
    
    return await document_controller.generate_meeting_document(
        doc_title=doc_title,
        doc_type="user_story_generator",
        doc_format=doc_format,
        enable_missing_questions=enable_missing_questions,
        enable_process_map=enable_process_map,
        include_screenshots=include_screenshots,
        session_guid=session_guid,
        video_path=video_path,
        client_name=client_name,
        teams_llm_config=llm_params
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "MDoc API"}
import os
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from src.orchestrator.workflow import workflow_engine
from src.data_pipeline.file_processor import save_stream_to_temp

router = APIRouter(prefix="/api", tags=["Dataset Analysis"])
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


@router.post("/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    target_column: str = Form(None)
):
    """
    Upload CSV/Excel -> S3 -> analyze (small vs large)
    target_column required only for large datasets.
    """

    try:
        # Validate MIME types
        if not file.filename.lower().endswith((".csv", ".xls", ".xlsx")):
            raise HTTPException(
                status_code=400,
                detail="Only CSV, XLS, XLSX files are supported."
            )

        # Save upload into temp directory
        temp_filename = f"{uuid.uuid4().hex}_{file.filename}"
        local_path = save_stream_to_temp(file.file, filename=temp_filename)

        # Create S3 key (same name)
        s3_key = temp_filename

        # Run main analysis workflow
        result = workflow_engine.run(
            local_path=local_path,
            s3_key=s3_key,
            target_column=target_column
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": result
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

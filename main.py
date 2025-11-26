import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.handlers.error_handler import register_exception_handlers
from src.utils.logger import setup_logging

# ------------------------------------------
# INIT LOGGING
# ------------------------------------------
setup_logging()
logger = logging.getLogger("generative_ai_project")

# ------------------------------------------
# FASTAPI APP
# ------------------------------------------
app = FastAPI(
    title="Dataset Analysis & Forecasting Service",
    version="1.0.0",
    description="Uploads CSV/Excel → S3 → Stats → LLM or LSTM + LLM analysis."
)

# ------------------------------------------
# CORS (optional but safe for frontend)
# ------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------
# ROUTES
# ------------------------------------------
app.include_router(api_router)

# ------------------------------------------
# ERROR HANDLERS
# ------------------------------------------
register_exception_handlers(app)

# ------------------------------------------
# HEALTH CHECK
# ------------------------------------------
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "service running"}


# ------------------------------------------
# START LOG
# ------------------------------------------
logger.info("Application startup complete.")

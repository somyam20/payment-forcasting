"""
Processors Module
Organized into submodules:
- audio: Audio processing (Whisper, speech recognition, etc.)
- video: Video processing (frame extraction, screenshot detection)
- parallel: Parallel processing utilities
- utils: Processor utilities (chunking, PII protection, etc.)
"""

from .video.video_processor import VideoProcessor
from .video.screenshot_extractor import ScreenshotExtractor
from .audio.whisper_processor import WhisperProcessor, get_optimized_whisper_processor

__all__ = [
    'VideoProcessor',
    'ScreenshotExtractor',
    'WhisperProcessor',
    'get_optimized_whisper_processor',
]


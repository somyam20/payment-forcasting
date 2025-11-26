"""
Whisper Speech Processing Module

This module provides Whisper-based speech-to-text processing for better accuracy
compared to Google Speech Recognition.
"""

import whisper
import ffmpeg
import tempfile
import os
from pydub import AudioSegment
from typing import List, Tuple, Optional
import numpy as np
import threading
import psutil

# Global singleton instance
_whisper_processor_instance = None
_whisper_lock = threading.Lock()


class WhisperProcessor:
    """
    High-quality speech-to-text processing using OpenAI Whisper
    """
    
    def __init__(self, model_size="base"):
        """
        Initialize Whisper processor
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       - tiny: fastest, least accurate
                       - base: good balance of speed and accuracy
                       - small: better accuracy, slower
                       - medium: high accuracy, much slower
                       - large: highest accuracy, slowest
        """
        self.model_size = model_size
        self.model = None
        self._transcription_lock = threading.Lock()  # Thread-safe transcription
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            print(f"Loading Whisper model: {self.model_size}")
            # Use CPU to avoid GPU memory issues on most systems
            self.model = whisper.load_model(self.model_size, device="cpu")
            print(f"Whisper {self.model_size} model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            # Fallback to tiny model if requested model fails
            if self.model_size != "tiny":
                print("Falling back to tiny model")
                self.model_size = "tiny"
                self.model = whisper.load_model("tiny", device="cpu")
            else:
                raise e
    
    def convert_video_to_wav(self, video_path: str, output_path: str = None) -> str:
        """
        Convert video file to WAV audio format for Whisper processing
        
        Args:
            video_path: Path to input video file
            output_path: Path for output WAV file (optional)
            
        Returns:
            Path to the converted WAV file
        """
        if output_path is None:
            # Create temporary file
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "audio_for_whisper.wav")
        
        try:
            # Use pydub for reliable audio extraction
            print(f"Converting video to audio: {video_path}")
            audio = AudioSegment.from_file(video_path)
            
            # Convert to mono and resample to 16kHz (Whisper's preferred format)
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(16000)  # Resample to 16kHz
            
            # Export as WAV
            audio.export(output_path, format="wav")
            print(f"Audio conversion complete: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error converting video to audio: {e}")
            # Fallback to ffmpeg-python if pydub fails
            try:
                print("Trying ffmpeg-python as fallback")
                (
                    ffmpeg
                    .input(video_path)
                    .output(output_path, acodec='pcm_s16le', ac=1, ar='16000')
                    .overwrite_output()
                    .run(quiet=True)
                )
                return output_path
            except Exception as ffmpeg_error:
                raise Exception(f"Both pydub and ffmpeg failed: {e}, {ffmpeg_error}")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe audio file using Whisper (thread-safe)
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr') - auto-detect if None
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        if self.model is None:
            raise Exception("Whisper model not loaded")
        
        # Use lock to ensure thread-safe transcription
        with self._transcription_lock:
            try:
                print(f"Transcribing audio with Whisper {self.model_size} model")
                
                # Configure transcription options
                options = {
                    "task": "transcribe",
                    "verbose": False,
                    "word_timestamps": True,  # Enable word-level timestamps
                    "fp16": False  # Disable FP16 for CPU compatibility
                }
                
                # Only add language if specified
                if language:
                    options["language"] = language
                
                # Perform transcription
                result = self.model.transcribe(audio_path, **options)
                
                if result is None:
                    raise Exception("Whisper returned None result")
                
                print(f"Transcription complete. Found {len(result.get('segments', []))} segments")
                return result
                
            except Exception as e:
                print(f"Error during Whisper transcription: {e}")
                raise e
    
    def extract_speech_segments(self, video_path: str, language: str = None) -> List[Tuple[float, str]]:
        """
        Extract speech segments from video with timestamps
        
        Args:
            video_path: Path to video file
            language: Language code for transcription
            
        Returns:
            List of (timestamp, text) tuples
        """
        try:
            # Convert video to audio
            audio_path = self.convert_video_to_wav(video_path)
            
            # Transcribe audio
            result = self.transcribe_audio(audio_path, language)
            
            # Extract segments with timestamps
            speech_segments = []
            
            if 'segments' in result:
                for segment in result['segments']:
                    timestamp = segment.get('start', 0.0)
                    text = segment.get('text', '').strip()
                    
                    if text:  # Only add non-empty text
                        speech_segments.append((timestamp, text))
            
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            print(f"Extracted {len(speech_segments)} speech segments using Whisper")
            return speech_segments
            
        except Exception as e:
            print(f"Error extracting speech segments: {e}")
            return []
    
    def extract_speech_with_keywords(self, video_path: str, keywords: List[str] = None, language: str = None) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
        """
        Extract speech segments and identify keyword matches
        
        Args:
            video_path: Path to video file
            keywords: List of keywords to search for
            language: Language code for transcription
            
        Returns:
            Tuple of (all_speech_segments, keyword_segments)
        """
        # Get all speech segments
        speech_segments = self.extract_speech_segments(video_path, language)
        
        # Find keyword matches if keywords provided
        keyword_segments = []
        if keywords and speech_segments:
            keywords_lower = [kw.lower() for kw in keywords]
            
            for timestamp, text in speech_segments:
                text_lower = text.lower()
                for keyword in keywords_lower:
                    if keyword in text_lower:
                        keyword_segments.append((timestamp, f"Keyword '{keyword}': {text}"))
                        break  # Avoid duplicate entries for same segment
        
        return speech_segments, keyword_segments
    
    def get_detailed_transcription(self, video_path: str, language: str = None) -> dict:
        """
        Get detailed transcription with word-level timestamps
        
        Args:
            video_path: Path to video file
            language: Language code for transcription
            
        Returns:
            Detailed transcription data
        """
        try:
            # Convert video to audio
            audio_path = self.convert_video_to_wav(video_path)
            
            # Get full transcription with word timestamps
            result = self.transcribe_audio(audio_path, language)
            
            # Clean up
            try:
                os.remove(audio_path)
            except:
                pass
            
            # Extract detailed information
            detailed_result = {
                'full_text': result.get('text', ''),
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'duration': 0.0
            }
            
            # Calculate total duration
            if detailed_result['segments']:
                last_segment = detailed_result['segments'][-1]
                detailed_result['duration'] = last_segment.get('end', 0.0)
            
            return detailed_result
            
        except Exception as e:
            print(f"Error getting detailed transcription: {e}")
            return {
                'full_text': '',
                'language': 'unknown',
                'segments': [],
                'duration': 0.0
            }


def get_optimized_whisper_processor() -> WhisperProcessor:
    """
    Get an optimized Whisper processor based on system capabilities (singleton pattern)
    
    Returns:
        WhisperProcessor instance with appropriate model size
    """
    global _whisper_processor_instance
    
    # Use singleton pattern to prevent multiple model loading
    with _whisper_lock:
        if _whisper_processor_instance is None:
            try:
                # Try to detect system capabilities and choose appropriate model
                # Get available memory
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                if memory_gb >= 8:
                    model_size = "base"  # Good balance for most systems
                elif memory_gb >= 4:
                    model_size = "tiny"  # Faster for limited memory
                else:
                    model_size = "tiny"  # Safest for low-memory systems
                    
                print(f"System memory: {memory_gb:.1f}GB, using Whisper '{model_size}' model")
                _whisper_processor_instance = WhisperProcessor(model_size)
                
            except ImportError:
                # psutil not available, use conservative default
                print("Using Whisper 'base' model (default)")
                _whisper_processor_instance = WhisperProcessor("base")
            except Exception as e:
                print(f"Error detecting system capabilities: {e}, using tiny model")
                _whisper_processor_instance = WhisperProcessor("tiny")
        
        return _whisper_processor_instance


# Convenience function for direct usage
def transcribe_video_with_whisper(video_path: str, keywords: List[str] = None, language: str = None) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]:
    """
    Convenience function to transcribe video using optimized Whisper processor
    
    Args:
        video_path: Path to video file
        keywords: Optional list of keywords to detect
        language: Optional language code
        
    Returns:
        Tuple of (speech_segments, keyword_segments)
    """
    processor = get_optimized_whisper_processor()
    return processor.extract_speech_with_keywords(video_path, keywords, language)
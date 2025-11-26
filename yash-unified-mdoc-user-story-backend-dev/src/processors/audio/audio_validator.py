"""
Audio Validation and Preprocessing Module

This module provides robust audio validation and preprocessing to prevent
Whisper tensor reshape errors and ensure reliable transcription processing.
"""

import os
import tempfile
import numpy as np
import wave
from pydub import AudioSegment
from typing import Optional, Tuple, List
import librosa

from ...utils.logger_config import setup_logger

setup_logger()


class AudioValidator:
    """
    Comprehensive audio validation and preprocessing for Whisper
    """
    
    @staticmethod
    def validate_audio_data(raw_data: bytes, sample_rate: int = 16000) -> bool:
        """
        Validate raw audio data before processing
        
        Args:
            raw_data: Raw audio bytes
            sample_rate: Expected sample rate
            
        Returns:
            True if audio is valid for processing
        """
        try:
            # Check minimum data size (at least 0.1 seconds)
            min_size = int(sample_rate * 0.1 * 2)  # 16-bit = 2 bytes per sample
            if len(raw_data) < min_size:
                return False
            
            # Check for valid audio content (not all zeros)
            audio_array = np.frombuffer(raw_data, dtype=np.int16)
            if np.all(audio_array == 0):
                return False
            
            # Check for reasonable amplitude range
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude < 10:  # Very quiet audio
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def create_valid_wav_file(raw_data: bytes, sample_rate: int = 16000, 
                            channels: int = 1) -> Optional[str]:
        """
        Create a valid WAV file from raw audio data with comprehensive validation
        
        Args:
            raw_data: Raw audio bytes
            sample_rate: Sample rate (default 16kHz for Whisper)
            channels: Number of channels (default 1 for mono)
            
        Returns:
            Path to created WAV file, or None if invalid
        """
        try:
            # Validate input data
            if not AudioValidator.validate_audio_data(raw_data, sample_rate):
                return None
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Write WAV file with proper format
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(raw_data)
            
            # Verify the created file
            if os.path.getsize(temp_file.name) < 1000:
                os.unlink(temp_file.name)
                return None
            
            # Additional validation using librosa
            try:
                audio, sr = librosa.load(temp_file.name, sr=sample_rate)
                if len(audio) < sample_rate * 0.1:  # Less than 0.1 seconds
                    os.unlink(temp_file.name)
                    return None
                    
                # Check for valid audio signal
                if np.max(np.abs(audio)) < 0.001:  # Very quiet
                    os.unlink(temp_file.name)
                    return None
                    
            except Exception:
                # If librosa fails, fall back to basic validation
                pass
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error creating WAV file: {e}")
            return None
    
    @staticmethod
    def preprocess_audio_segment(audio_segment: AudioSegment) -> Optional[str]:
        """
        Preprocess AudioSegment for optimal Whisper processing
        
        Args:
            audio_segment: Input audio segment
            
        Returns:
            Path to preprocessed WAV file, or None if invalid
        """
        try:
            # Validate segment length
            if len(audio_segment) < 500:  # Less than 0.5 seconds
                return None
            
            # Check audio level
            if audio_segment.dBFS < -50:  # Very quiet
                return None
            
            # Optimize for Whisper
            optimized = audio_segment.set_channels(1)  # Mono
            optimized = optimized.set_frame_rate(16000)  # 16kHz
            optimized = optimized.set_sample_width(2)  # 16-bit
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Export with optimal parameters
            optimized.export(
                temp_file.name,
                format="wav",
                parameters=[
                    "-ac", "1",           # Mono
                    "-ar", "16000",       # 16kHz sample rate
                    "-acodec", "pcm_s16le"  # 16-bit PCM
                ]
            )
            
            # Validate output file
            if os.path.getsize(temp_file.name) < 1000:
                os.unlink(temp_file.name)
                return None
            
            # Final validation with librosa
            try:
                audio, sr = librosa.load(temp_file.name, sr=16000)
                if len(audio) < 0.1 * 16000 or np.max(np.abs(audio)) < 0.001:
                    os.unlink(temp_file.name)
                    return None
            except Exception:
                pass
            
            return temp_file.name
            
        except Exception as e:
            print(f"Error preprocessing audio segment: {e}")
            return None
    
    @staticmethod
    def validate_wav_file(file_path: str) -> bool:
        """
        Validate a WAV file for Whisper compatibility
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            True if file is valid for Whisper processing
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check file size
            if os.path.getsize(file_path) < 1000:
                return False
            
            # Validate with wave module
            with wave.open(file_path, 'rb') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                
                # Check basic properties
                if frames < sample_rate * 0.1:  # Less than 0.1 seconds
                    return False
                if channels > 2:  # Too many channels
                    return False
                if sample_rate < 8000:  # Too low sample rate
                    return False
            
            # Validate with librosa for audio content
            try:
                audio, sr = librosa.load(file_path, sr=None)
                if len(audio) == 0 or np.max(np.abs(audio)) < 0.001:
                    return False
            except Exception:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def create_whisper_compatible_audio(input_path: str) -> Optional[str]:
        """
        Create a Whisper-compatible audio file from any input audio
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Path to Whisper-compatible WAV file, or None if conversion failed
        """
        try:
            # Load audio with librosa for robust handling
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            
            # Validate loaded audio
            if len(audio) < 0.1 * 16000:  # Less than 0.1 seconds
                return None
            
            if np.max(np.abs(audio)) < 0.001:  # Very quiet
                return None
            
            # Normalize audio to prevent clipping
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Convert to 16-bit integer
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create output file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_file.close()
            
            # Write WAV file
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            
            # Final validation
            if AudioValidator.validate_wav_file(temp_file.name):
                return temp_file.name
            else:
                os.unlink(temp_file.name)
                return None
                
        except Exception as e:
            print(f"Error creating Whisper-compatible audio: {e}")
            return None


def safe_whisper_transcribe(whisper_processor, audio_input, input_type="file"):
    """
    Safely transcribe audio with comprehensive validation and preprocessing
    
    Args:
        whisper_processor: WhisperProcessor instance
        audio_input: Audio input (file path, raw data, or AudioSegment)
        input_type: Type of input ("file", "raw_data", "audio_segment")
        
    Returns:
        Transcription result dict, or empty dict if processing failed
    """
    temp_files = []
    
    try:
        validated_file = None
        
        if input_type == "file":
            # Validate existing file
            if AudioValidator.validate_wav_file(audio_input):
                validated_file = audio_input
            else:
                # Try to create compatible version
                validated_file = AudioValidator.create_whisper_compatible_audio(audio_input)
                if validated_file:
                    temp_files.append(validated_file)
        
        elif input_type == "raw_data":
            # Create WAV from raw data
            validated_file = AudioValidator.create_valid_wav_file(audio_input)
            if validated_file:
                temp_files.append(validated_file)
        
        elif input_type == "audio_segment":
            # Preprocess AudioSegment
            validated_file = AudioValidator.preprocess_audio_segment(audio_input)
            if validated_file:
                temp_files.append(validated_file)
        
        # Perform transcription if we have valid audio
        if validated_file and os.path.exists(validated_file):
            print(f"Transcribing validated audio file: {os.path.getsize(validated_file)} bytes")
            result = whisper_processor.transcribe_audio(validated_file)
            return result
        else:
            print("Audio validation failed - skipping transcription")
            return {"text": "", "segments": []}
    
    except Exception as e:
        print(f"Safe transcription failed: {e}")
        return {"text": "", "segments": []}
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
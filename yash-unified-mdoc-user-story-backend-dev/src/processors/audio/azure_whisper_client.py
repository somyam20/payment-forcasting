"""
Direct Azure OpenAI Whisper Integration

This module provides direct Azure OpenAI Whisper integration within the main application,
eliminating the need for a separate service while maintaining superior accuracy.
"""

import os
import tempfile
import time
from typing import Optional, Dict, Any, List, Tuple
from pydub import AudioSegment
import logging
from openai import OpenAI, AzureOpenAI

logger = logging.getLogger(__name__)

from ...utils.logger_config import setup_logger
from dotenv import load_dotenv
load_dotenv()

setup_logger()

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Replace with your Key Vault URL
key_vault_url = os.getenv('KEY_VAULT_URL')

tenant_id = os.getenv('AZURE_TENANT_ID')
credential = DefaultAzureCredential(
    additionally_allowed_tenants=[tenant_id] if tenant_id else []
)
# Authenticate using DefaultAzureCredential (supports Managed Identity, CLI login, etc.)
# credential = DefaultAzureCredential()
client = SecretClient(vault_url=key_vault_url, credential=credential)

class DirectAzureWhisperClient:
    """
    Direct Azure OpenAI Whisper client for in-application speech processing
    """
    
    def __init__(self):
        """Initialize Azure OpenAI Whisper client"""
        self.client = None
        self.service_type = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup Azure OpenAI or standard OpenAI client"""
        try:
            # Try Azure OpenAI first
            azure_endpoint = client.get_secret('AZURE_OPENAI_ENDPOINT').value
            azure_key = client.get_secret('AZURE_OPENAI_API_KEY').value
            azure_version = client.get_secret('AZURE_OPENAI_API_VERSION', '2024-02-01').value
            
            if azure_endpoint and azure_key:
                self.client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=azure_version
                )
                self.service_type = "Azure OpenAI"
                logging.info("Azure OpenAI Whisper client initialized")
            else:
                # Fallback to standard OpenAI
                openai_key = client.get_secret('OPENAI_API_KEY').value
                if openai_key:
                    self.client = OpenAI(api_key=openai_key)
                    self.service_type = "Standard OpenAI"
                    logging.info("Standard OpenAI Whisper client initialized")
                else:
                    logging.warning("No OpenAI credentials found - speech recognition will use fallbacks")
                    
        except Exception as e:
            logging.error(f"Failed to initialize Whisper client: {e}")
    
    def is_available(self) -> bool:
        """Check if Whisper client is available"""
        return self.client is not None
    
    def transcribe_audio_segment(self, audio_segment: AudioSegment, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe AudioSegment using Azure/Standard OpenAI Whisper
        
        Args:
            audio_segment: pydub AudioSegment
            language: Optional language code
            
        Returns:
            Transcription result dictionary
        """
        if not self.is_available():
            raise Exception("Whisper client not available")
        
        # Create temporary file for audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        try:
            # Export as optimized WAV for Whisper
            audio_segment.set_channels(1).set_frame_rate(16000).export(
                temp_file.name, format="wav"
            )
            
            return self._transcribe_file(temp_file.name, language)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def transcribe_audio_data(self, audio_data: bytes, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe raw audio data using Azure/Standard OpenAI Whisper
        
        Args:
            audio_data: Raw audio data in bytes
            language: Optional language code
            
        Returns:
            Transcription result dictionary
        """
        if not self.is_available():
            raise Exception("Whisper client not available")
        
        # Create temporary file for audio data
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        try:
            # Write audio data to file
            with open(temp_file.name, 'wb') as f:
                f.write(audio_data)
            
            return self._transcribe_file(temp_file.name, language)
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def _transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Azure/Standard OpenAI Whisper
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            
        Returns:
            Transcription result dictionary
        """
        start_time = time.time()
        
        try:
            logging.info(f"Transcribing with {self.service_type} Whisper")
            
            with open(file_path, 'rb') as audio_file:
                # Try with word-level timestamps first
                try:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        language=language if language else None,
                        timestamp_granularities=["word"] if hasattr(self.client.audio.transcriptions, 'create') else None
                    )
                except Exception:
                    # Fallback without word-level timestamps
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="verbose_json",
                        language=language if language else None
                    )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logging.info(f"{self.service_type} Whisper transcription completed in {processing_time:.2f}s")
            
            # Convert to application-compatible format
            result = {
                'text': transcript.text,
                'segments': [],
                'language': getattr(transcript, 'language', 'en'),
                'duration': getattr(transcript, 'duration', 0.0),
                'processing_time': processing_time,
                'service_type': self.service_type
            }
            
            # Add word-level timestamps if available
            if hasattr(transcript, 'words') and transcript.words:
                result['segments'] = [
                    {
                        'start': word.start,
                        'end': word.end,
                        'text': word.word
                    }
                    for word in transcript.words
                ]
            elif hasattr(transcript, 'segments') and transcript.segments:
                result['segments'] = [
                    {
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text
                    }
                    for seg in transcript.segments
                ]
            
            return result
            
        except Exception as e:
            logging.error(f"{self.service_type} Whisper transcription error: {e}")
            raise
    
    def transcribe_audio_chunks_batch(self, audio_chunks: List[Tuple[float, AudioSegment]], 
                                    language: Optional[str] = None) -> List[Tuple[float, str]]:
        """
        Transcribe multiple audio chunks
        
        Args:
            audio_chunks: List of (timestamp, audio_segment) tuples
            language: Optional language code
            
        Returns:
            List of (timestamp, transcribed_text) tuples
        """
        if not self.is_available():
            raise Exception("Whisper client not available")
        
        results = []
        
        for i, (timestamp, audio_segment) in enumerate(audio_chunks):
            try:
                logging.info(f"Transcribing chunk {i+1}/{len(audio_chunks)} with {self.service_type} Whisper")
                
                start_time = time.time()
                result = self.transcribe_audio_segment(audio_segment, language)
                end_time = time.time()
                
                text = result.get('text', '').strip()
                results.append((timestamp, text))
                
                logging.info(f"Chunk {i+1} completed in {end_time - start_time:.2f}s")
                
            except Exception as e:
                logging.error(f"Chunk {i+1} failed: {e}")
                results.append((timestamp, ""))  # Empty text for failed chunks
        
        return results


# Global client instance
_azure_whisper_client = None

def get_azure_whisper_client() -> DirectAzureWhisperClient:
    """
    Get global Azure Whisper client instance
    
    Returns:
        DirectAzureWhisperClient instance
    """
    global _azure_whisper_client
    
    if _azure_whisper_client is None:
        _azure_whisper_client = DirectAzureWhisperClient()
    
    return _azure_whisper_client


def transcribe_with_azure_whisper(audio_chunks: List[Tuple[float, AudioSegment]], 
                                 language: Optional[str] = None) -> List[Tuple[float, str]]:
    """
    Convenience function for transcribing audio chunks with Azure OpenAI Whisper
    
    Args:
        audio_chunks: List of (timestamp, audio_segment) tuples
        language: Optional language code
        
    Returns:
        List of (timestamp, transcribed_text) tuples
    """
    client = get_azure_whisper_client()
    return client.transcribe_audio_chunks_batch(audio_chunks, language)
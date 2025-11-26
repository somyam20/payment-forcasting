"""
Azure AI Speech Integration

This module provides Azure AI Speech integration for speech-to-text processing,
offering unlimited rate limits and excellent accuracy as the primary speech service.
Supports both SDK and REST API approaches for maximum compatibility.
"""

import os
import logging
import tempfile
import time
import threading
import json
import requests
from typing import Dict, Any, Optional, List, Tuple
from pydub import AudioSegment
import logging

from ...utils.logger_config import setup_logger

setup_logger()

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# Replace with your Key Vault URL
key_vault_url = os.getenv('KEY_VAULT_URL')

tenant_id = os.getenv('AZURE_TENANT_ID')
credential = DefaultAzureCredential(
    additionally_allowed_tenants=[tenant_id] if tenant_id else []
)
# credential = DefaultAzureCredential()
vault_client = SecretClient(vault_url=key_vault_url, credential=credential)


# Try to import Azure Speech SDK, fall back to REST API if not available
try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_SPEECH_SDK_AVAILABLE = True
except Exception as e:
    AZURE_SPEECH_SDK_AVAILABLE = False
    speechsdk = None
    logging.exception(f"Azure Speech SDK not available, using REST API: {e}")

class AzureSpeechClient:
    """
    Azure AI Speech client for speech-to-text processing with no rate limits
    Supports both SDK and REST API approaches for maximum compatibility
    """
    
    def __init__(self):
        """Initialize Azure AI Speech client"""
        self.speech_config = None
        self.service_type = None
        self.speech_key = None
        self.speech_region = None
        self.speech_endpoint = None
        self.use_sdk = False
        self._setup_client()
    
    def _setup_client(self):
        """Setup Azure AI Speech client"""
        try:
            # Get Azure AI Speech credentials
            self.speech_key = vault_client.get_secret('AZURE-SPEECH-KEY').value
            self.speech_region = vault_client.get_secret('AZURE-SPEECH-REGION').value
            
            if not self.speech_key or not self.speech_region:
                logging.warning("Azure AI Speech credentials not found")
                self.service_type = None
                return
            
            # Try SDK first if available
            if AZURE_SPEECH_SDK_AVAILABLE:
                try:
                    self.speech_config = speechsdk.SpeechConfig(
                        subscription=self.speech_key, 
                        region=self.speech_region
                    )
                    self.speech_config.speech_recognition_language = "en-IN"
                    self.use_sdk = True
                    self.service_type = "Azure AI Speech SDK"
                    logging.info("Azure AI Speech SDK client initialized")
                    return
                except Exception as sdk_error:
                    logging.warning(f"SDK initialization failed: {sdk_error}. Using REST API.")
            
            # Fall back to REST API
            self.speech_endpoint = f"https://{self.speech_region}.api.cognitive.microsoft.com/"
            self.use_sdk = False
            self.service_type = "Azure AI Speech REST"
            logging.info("Azure AI Speech REST client initialized")
                
        except Exception as e:
            logging.error(f"Failed to initialize Azure AI Speech client: {e}")
            self.service_type = None
    
    def is_available(self) -> bool:
        """Check if Azure AI Speech client is available"""
        return self.speech_key is not None and self.speech_region is not None and self.service_type is not None
    
    def transcribe_audio_segment(self, audio_segment: AudioSegment, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe AudioSegment using Azure AI Speech
        
        Args:
            audio_segment: pydub AudioSegment
            language: Optional language code (e.g., 'en-IN', 'es-ES')
            
        Returns:
            Transcription result dictionary
        """
        if not self.is_available():
            raise Exception("Azure AI Speech client not available")
        
        # Create temporary file for audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        
        try:
            # Export audio to WAV format (Azure Speech prefers WAV)
            audio_segment.export(temp_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            
            # Transcribe using appropriate method
            if self.use_sdk:
                result = self._transcribe_file_sdk(temp_file.name, language)
            else:
                result = self._transcribe_file_rest(temp_file.name, language)
            return result
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def transcribe_audio_data(self, audio_data: bytes, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe raw audio data using Azure AI Speech
        
        Args:
            audio_data: Raw audio data in bytes
            language: Optional language code
            
        Returns:
            Transcription result dictionary
        """
        import io
        audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
        return self.transcribe_audio_segment(audio_segment, language)
    
    def _transcribe_file_sdk(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Azure AI Speech SDK
        """
        try:
            # Set language if provided
            if language:
                self.speech_config.speech_recognition_language = language
            
            # Create audio configuration from file
            audio_config = speechsdk.AudioConfig(filename=file_path)
            
            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # Collect all transcribed text using continuous recognition
            all_text = []
            done = threading.Event()
            
            def handle_final_result(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    all_text.append(evt.result.text)
                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logging.debug("No speech could be recognized")
                elif evt.result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speechsdk.CancellationDetails(evt.result)
                    logging.error(f"Speech recognition canceled: {cancellation_details.reason}")
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        logging.error(f"Error details: {cancellation_details.error_details}")
            
            def handle_session_stopped(evt):
                done.set()
            
            # Connect event handlers
            recognizer.recognized.connect(handle_final_result)
            recognizer.session_stopped.connect(handle_session_stopped)
            recognizer.canceled.connect(handle_session_stopped)
            
            # Start continuous recognition
            recognizer.start_continuous_recognition()
            
            # Wait for completion with timeout
            if not done.wait(timeout=60):  # 60 second timeout
                logging.warning("Speech recognition timed out")
            
            # Stop recognition
            recognizer.stop_continuous_recognition()
            
            # Return complete text
            complete_text = " ".join(all_text).strip()
            
            return {
                'text': complete_text,
                'confidence': 0.95,
                'service': 'Azure AI Speech SDK',
                'language': language or 'en-IN'
            }
                    
        except Exception as e:
            logging.error(f"Azure AI Speech SDK transcription failed: {e}")
            raise e
    
    def _transcribe_file_rest(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Azure AI Speech REST API
        """
        try:
            # Use direct subscription key instead of token for simplicity
            url = f"https://{self.speech_region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            # Parameters
            params = {
                'language': language or 'en-IN',
                'format': 'detailed'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.speech_key,
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                'Accept': 'application/json'
            }
            
            # Read audio file
            with open(file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Make request
            response = requests.post(url, params=params, headers=headers, data=audio_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Handle different response formats
                text = ''
                if 'DisplayText' in result:
                    text = result['DisplayText']
                elif 'NBest' in result and len(result['NBest']) > 0:
                    text = result['NBest'][0].get('Display', '')
                elif 'RecognitionStatus' in result and result['RecognitionStatus'] == 'Success':
                    text = result.get('DisplayText', '')
                
                return {
                    'text': text,
                    'confidence': 0.9,
                    'service': 'Azure AI Speech REST',
                    'language': language or 'en-IN'
                }
            else:
                raise Exception(f"Azure AI Speech REST API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logging.error(f"Azure AI Speech REST transcription failed: {e}")
            raise e
    
    def _get_access_token(self) -> Optional[str]:
        """
        Get access token for Azure AI Speech REST API
        """
        try:
            token_url = f"https://{self.speech_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
            headers = {
                'Ocp-Apim-Subscription-Key': self.speech_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(token_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.text
            else:
                logging.error(f"Failed to get access token: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting access token: {e}")
            return None
    
    def transcribe_audio_chunks_batch(self, audio_chunks: List[Tuple[float, AudioSegment]], 
                                    language: Optional[str] = None) -> List[Tuple[float, str]]:
        """
        Transcribe multiple audio chunks with Azure AI Speech
        
        Args:
            audio_chunks: List of (timestamp, audio_segment) tuples
            language: Optional language code
            
        Returns:
            List of (timestamp, transcribed_text) tuples
        """
        results = []
        
        for timestamp, audio_segment in audio_chunks:
            try:
                result = self.transcribe_audio_segment(audio_segment, language)
                text = result.get('text', '')
                results.append((timestamp, text))
                logging.info(f"Azure AI Speech transcribed chunk at {timestamp}s: {len(text)} chars")
            except Exception as e:
                logging.error(f"Azure AI Speech failed for chunk at {timestamp}s: {e}")
                results.append((timestamp, ''))
        
        return results

# Global client instance
_azure_speech_client = None

def get_azure_speech_client() -> AzureSpeechClient:
    """
    Get global Azure AI Speech client instance
    
    Returns:
        AzureSpeechClient instance
    """
    global _azure_speech_client
    if _azure_speech_client is None:
        _azure_speech_client = AzureSpeechClient()
    return _azure_speech_client

def transcribe_with_azure_speech(audio_chunks: List[Tuple[float, AudioSegment]], 
                                language: Optional[str] = None) -> List[Tuple[float, str]]:
    """
    Convenience function for transcribing audio chunks with Azure AI Speech
    
    Args:
        audio_chunks: List of (timestamp, audio_segment) tuples
        language: Optional language code
        
    Returns:
        List of (timestamp, transcribed_text) tuples
    """
    client = get_azure_speech_client()
    if not client.is_available():
        logging.warning("Azure AI Speech not available, returning empty results")
        return [(timestamp, '') for timestamp, _ in audio_chunks]
    
    return client.transcribe_audio_chunks_batch(audio_chunks, language)
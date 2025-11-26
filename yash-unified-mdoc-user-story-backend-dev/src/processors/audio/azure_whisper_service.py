"""
Azure OpenAI Whisper Service

This service uses Azure OpenAI's Whisper API for speech-to-text processing,
providing superior accuracy and eliminating local processing overhead.
"""

import os
import tempfile
import time
import base64
import uuid
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify
import logging
from openai import AzureOpenAI
from pydub import AudioSegment

import logging

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
# credential = DefaultAzureCredential()
vault_client = SecretClient(vault_url=key_vault_url, credential=credential)

class AzureWhisperService:
    """
    Azure OpenAI Whisper transcription service
    """
    
    def __init__(self, port=5001):
        """
        Initialize Azure Whisper service
        
        Args:
            port: Service port
        """
        self.port = port
        self.client = None
        self.app = Flask(__name__)
        self._setup_azure_client()
        self._setup_routes()
        
    def _setup_azure_client(self):
        """Setup Azure OpenAI client"""
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
                openai_key = vault_client.get_secret('OPENAI-API-KEY').value
                if openai_key:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=openai_key)
                    self.service_type = "OpenAI"
                    logging.info("Standard OpenAI Whisper client initialized")
                else:
                    raise Exception("No OpenAI credentials found")
                    
        except Exception as e:
            logging.error(f"Failed to initialize Whisper client: {e}")
            raise
    
    def _setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service_type': self.service_type,
                'ready': self.client is not None
            })
        
        @self.app.route('/transcribe', methods=['POST'])
        def transcribe_audio():
            """Transcribe audio endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'audio_data' not in data:
                    return jsonify({'error': 'Missing audio_data'}), 400
                
                # Decode base64 audio data
                audio_data = base64.b64decode(data['audio_data'])
                language = data.get('language', None)
                
                # Generate unique filename
                temp_id = str(uuid.uuid4())
                temp_path = os.path.join(tempfile.gettempdir(), f"azure_whisper_temp_{temp_id}.wav")
                
                try:
                    # Save audio data to temporary file
                    with open(temp_path, 'wb') as f:
                        f.write(audio_data)
                    
                    # Transcribe with Azure OpenAI Whisper
                    result = self._transcribe_file(temp_path, language)
                    
                    return jsonify({
                        'success': True,
                        'transcription': result
                    })
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except Exception as e:
                logging.error(f"Transcription error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/transcribe_file', methods=['POST'])
        def transcribe_file():
            """Transcribe audio file endpoint"""
            try:
                if 'audio_file' not in request.files:
                    return jsonify({'error': 'No audio file provided'}), 400
                
                file = request.files['audio_file']
                language = request.form.get('language', None)
                
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                # Generate unique filename
                temp_id = str(uuid.uuid4())
                temp_path = os.path.join(tempfile.gettempdir(), f"azure_whisper_upload_{temp_id}.wav")
                
                try:
                    # Save uploaded file
                    file.save(temp_path)
                    
                    # Transcribe with Azure OpenAI Whisper
                    result = self._transcribe_file(temp_path, language)
                    
                    return jsonify({
                        'success': True,
                        'transcription': result
                    })
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except Exception as e:
                logging.error(f"File transcription error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file using Azure OpenAI Whisper
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            
        Returns:
            Transcription result dictionary
        """
        if self.client is None:
            raise Exception("Azure OpenAI Whisper client not initialized")
        
        start_time = time.time()
        
        try:
            # Preprocess audio for optimal Whisper performance
            try:
                # Convert to optimal format for Whisper
                audio = AudioSegment.from_file(file_path)
                
                # Ensure mono channel and appropriate sample rate
                audio = audio.set_channels(1).set_frame_rate(16000)
                
                # Check for valid audio content
                if len(audio) == 0:
                    raise Exception("Audio file is empty")
                
                if len(audio) < 100:  # Less than 100ms
                    logging.warning(f"Very short audio segment: {len(audio)}ms")
                
                # Export to optimized WAV file
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_file.close()
                
                audio.export(temp_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                processed_file = temp_file.name
                
            except Exception as audio_error:
                logging.warning(f"Audio preprocessing failed, using original: {audio_error}")
                processed_file = file_path
            
            # Transcribe using Azure OpenAI Whisper API
            logging.info(f"Transcribing with {self.service_type} Whisper API")
            
            with open(processed_file, 'rb') as audio_file:
                # Prepare transcription parameters
                transcription_params = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json"
                }
                
                # Add language if specified
                if language:
                    transcription_params["language"] = language
                
                # Add timestamp granularities for word-level timestamps (if supported)
                try:
                    transcription_params["timestamp_granularities"] = ["word"]
                    transcript = self.client.audio.transcriptions.create(**transcription_params)
                except Exception as e:
                    # Fallback without word-level timestamps if not supported
                    logging.warning(f"Word-level timestamps not supported, using segment-level: {e}")
                    transcription_params.pop("timestamp_granularities", None)
                    transcript = self.client.audio.transcriptions.create(**transcription_params)
            
            # Clean up temporary file if created
            if processed_file != file_path:
                try:
                    os.unlink(processed_file)
                except:
                    pass
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logging.info(f"Azure Whisper transcription completed in {processing_time:.2f}s")
            
            # Convert Azure OpenAI response to compatible format
            result = {
                'text': transcript.text,
                'segments': [],
                'language': getattr(transcript, 'language', 'en'),
                'duration': getattr(transcript, 'duration', 0.0),
                'processing_time': str(processing_time),
                'service_type': self.service_type,
                'service_version': '1.0.0'
            }
            
            # Add word-level timestamps if available
            if hasattr(transcript, 'words') and transcript.words:
                segments = []
                for word in transcript.words:
                    segments.append({
                        'start': word.start,
                        'end': word.end,
                        'text': word.word
                    })
                result['segments'] = segments
            elif hasattr(transcript, 'segments') and transcript.segments:
                # If segments are available
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
            logging.error(f"Azure Whisper transcription error: {e}")
            raise
    
    def run(self, host='0.0.0.0', debug=False):
        """
        Start the Azure Whisper service
        
        Args:
            host: Host to bind to
            debug: Enable debug mode
        """
        logging.info(f"Starting Azure Whisper service on {host}:{self.port}")
        logging.info(f"Service type: {self.service_type}")
        
        self.app.run(
            host=host,
            port=self.port,
            debug=debug,
            threaded=True
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure OpenAI Whisper Transcription Service")
    parser.add_argument('--port', type=int, default=5001, help="Service port")
    parser.add_argument('--host', default='0.0.0.0', help="Host to bind to")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and start service
    service = AzureWhisperService(port=args.port)
    service.run(host=args.host, debug=args.debug)
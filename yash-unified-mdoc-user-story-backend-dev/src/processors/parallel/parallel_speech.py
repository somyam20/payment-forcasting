"""
Parallel speech processing module.

This module provides functions for processing speech segments in parallel,
significantly speeding up speech recognition and transcript analysis.
"""

import concurrent.futures
import os
import time
import io
import re
import tempfile
import speech_recognition as sr
from typing import List, Tuple, Dict, Any, Optional
import logging

from ...utils.logger_config import setup_logger

setup_logger()

def process_speech_chunk(chunk_data):
    """
    Process a single speech chunk using speech recognition.
    
    Args:
        chunk_data: Tuple containing (chunk_index, chunk_time, chunk, 
                                     total_chunks, recognizer, 
                                     trigger_patterns, use_ai)
                                     
    Returns:
        Tuple containing (speech_result, keyword_result)
    """
    chunk_index, chunk_time, chunk, total_chunks, recognizer, trigger_patterns, use_ai = chunk_data
    
    # Initialize results
    speech_result = None
    keyword_result = None
    
    # Convert chunk to format needed by SpeechRecognition
    chunk_file = io.BytesIO()
    chunk.export(chunk_file, format="wav")
    chunk_file.seek(0)
    
    try:
        # Use speech recognition to convert audio to text
        with sr.AudioFile(chunk_file) as source:
            audio_data = recognizer.record(source)
            
            # Try with speech recognition services - Whisper first for best accuracy
            text = ""
            
            # First try Azure AI Speech (no rate limits)
            try:
                from ..audio.azure_speech_client import get_azure_speech_client
                from pydub import AudioSegment
                
                # Convert SpeechRecognition audio data to AudioSegment
                audio_bytes = audio_data.get_raw_data()
                audio_segment = AudioSegment(
                    data=audio_bytes,
                    sample_width=2,  # 16-bit
                    frame_rate=16000,  # 16kHz
                    channels=1  # Mono
                )
                
                # Use Azure AI Speech directly
                azure_speech_client = get_azure_speech_client()
                if azure_speech_client.is_available():
                    result = azure_speech_client.transcribe_audio_segment(audio_segment)
                    text = result.get('text', '').strip()
                    logging.info(f"Chunk {chunk_index+1}/{total_chunks}: Recognized speech with Azure AI Speech")
                else:
                    raise Exception("Azure AI Speech not available")
                    
            except Exception as azure_speech_err:
                logging.exception(f"Azure AI Speech failed for chunk {chunk_index+1}: {azure_speech_err}")
                
                # Fallback to Azure OpenAI Whisper
                try:
                    from ..audio.azure_whisper_client import get_azure_whisper_client
                    from pydub import AudioSegment
                    
                    # Convert SpeechRecognition audio data to AudioSegment
                    audio_bytes = audio_data.get_raw_data()
                    audio_segment = AudioSegment(
                        data=audio_bytes,
                        sample_width=2,  # 16-bit
                        frame_rate=16000,  # 16kHz
                        channels=1  # Mono
                    )
                    
                    # Use Azure OpenAI Whisper directly
                    azure_client = get_azure_whisper_client()
                    if azure_client.is_available():
                        result = azure_client.transcribe_audio_segment(audio_segment)
                        text = result.get('text', '').strip()
                        logging.info(f"Chunk {chunk_index+1}/{total_chunks}: Recognized speech with {azure_client.service_type} Whisper")
                    else:
                        raise Exception("Azure OpenAI Whisper not available")
                        
                except Exception as whisper_local_err:
                    logging.exception(f"Azure OpenAI Whisper failed for chunk {chunk_index+1}: {whisper_local_err}")
                
                # Fallback to Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                    logging.info(f"Chunk {chunk_index+1}/{total_chunks}: Recognized speech with Google (fallback)")
                except Exception as google_err:
                    logging.exception(f"Google Speech Recognition failed: {google_err}")
                    
                    # Final fallback to OpenAI Whisper API if available
                    if use_ai:
                        try:
                            # Export chunk to a temporary file for Whisper API
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_file.close()
                            
                            # Save audio data
                            import wave
                            with wave.open(temp_file.name, 'wb') as wf:
                                wf.setnchannels(1)  # Mono
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(16000)  # 16kHz
                                wf.writeframes(audio_data.get_raw_data())
                            
                            # Call OpenAI Whisper API
                            from ...utils.media_utils import transcribe_with_whisper
                            text = transcribe_with_whisper(temp_file.name)
                            logging.info(f"Chunk {chunk_index+1}/{total_chunks}: Recognized speech with Whisper API (final fallback)")
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                        except Exception as whisper_api_err:
                            logging.exception(f"All speech recognition methods failed for chunk {chunk_index+1}: {whisper_api_err}")
            
            # Store transcript with timestamp
            if text:
                speech_result = (chunk_time, text)
                
                # Check for keywords in the transcript
                if trigger_patterns:
                    for pattern in trigger_patterns:
                        if re.search(pattern, text.lower()):
                            logging.info(f"Keyword found at {chunk_time:.2f}s: {text[:120]}")
                            keyword_result = (chunk_time, f"Keyword trigger: {pattern}")
                            break
    except Exception as e:
        logging.exception(f"Error in speech recognition for chunk {chunk_index+1}: {e}")
    
    return speech_result, keyword_result


def process_speech_in_parallel(audio_chunks, recognizer, trigger_patterns=None, use_ai=False):
    """
    Process speech chunks in parallel using multiple threads.
    Uses direct_parallel module for optimized performance.
    
    Args:
        audio_chunks: List of tuples (chunk_time, audio_chunk)
        recognizer: SpeechRecognition recognizer instance
        trigger_patterns: Optional list of regex patterns to match
        use_ai: Whether to use AI for speech recognition
        
    Returns:
        Tuple of (speech_results, keyword_results)
    """
    # Import the direct parallel implementation
    from .direct_parallel import parallel_speech_recognition
    
    # Use the highly optimized direct parallel implementation
    logging.info(f"Using optimized parallel speech recognition for {len(audio_chunks)} chunks")
    return parallel_speech_recognition(audio_chunks, recognizer, trigger_patterns, use_ai)
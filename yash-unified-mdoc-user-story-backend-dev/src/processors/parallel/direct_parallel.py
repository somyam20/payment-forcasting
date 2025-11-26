"""
Simple and direct parallel processing module.
This provides basic parallel execution functionality for speech recognition.
"""

import concurrent.futures
import os
import time
import tempfile
import speech_recognition as sr
from typing import List, Tuple, Any, Optional
import re
import logging


from ...utils.logger_config import setup_logger

setup_logger()

def process_speech_chunk(chunk_data):
    """
    Process a single speech chunk.
    
    Args:
        chunk_data: Tuple containing (index, timestamp, chunk, total_chunks, 
                                     recognizer, trigger_patterns, use_ai)
                                     
    Returns:
        Tuple containing (speech_result, keyword_result)
    """
    from ...utils.media_utils import transcribe_with_whisper
    
    index, timestamp, chunk, total_chunks, recognizer, trigger_patterns, use_ai = chunk_data
    
    # Initialize results
    speech_result = None
    keyword_result = None
    
    # Convert chunk to format needed by SpeechRecognition
    import io
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
                    logging.info(f"Chunk {index+1}/{total_chunks}: Recognized speech with Azure AI Speech")
                else:
                    raise Exception("Azure AI Speech not available")
                    
            except Exception as azure_speech_err:
                logging.exception(f"Azure AI Speech failed for chunk {index+1}: {azure_speech_err}")
                
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
                        logging.info(f"Chunk {index+1}/{total_chunks}: Recognized speech with {azure_client.service_type} Whisper")
                    else:
                        raise Exception("Azure OpenAI Whisper not available")
                        
                except Exception as whisper_service_err:
                    logging.exception(f"Azure OpenAI Whisper failed for chunk {index+1}: {whisper_service_err}")
                
                # Fallback to Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                    logging.info(f"Chunk {index+1}/{total_chunks}: Recognized speech with Google (fallback)")
                except Exception as google_err:
                    logging.exception(f"Google Speech Recognition failed: {google_err}")
                    
                    # Final fallback to OpenAI Whisper API if available
                    if use_ai:
                        try:
                            # Export chunk to a temporary file for Whisper API
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_file.close()
                            temp_path = temp_file.name
                            
                            # Save audio data
                            import wave
                            with wave.open(temp_path, 'wb') as wf:
                                wf.setnchannels(1)  # Mono
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(16000)  # 16kHz
                                wf.writeframes(audio_data.get_raw_data())
                            
                            # Call OpenAI Whisper API
                            text = transcribe_with_whisper(temp_path)
                            logging.info(f"Chunk {index+1}/{total_chunks}: Recognized speech with Whisper API (final fallback)")
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                        except Exception as whisper_api_err:
                            logging.exception(f"All speech recognition methods failed for chunk {index+1}: {whisper_api_err}")
            
            # Store transcript with timestamp
            if text:
                speech_result = (timestamp, text)
                
                # Check for keywords in the transcript
                if trigger_patterns:
                    for pattern in trigger_patterns:
                        if re.search(pattern, text.lower()):
                            keyword_result = (timestamp, f"Keyword trigger: {pattern}")
                            logging.info(f"Keyword found at {timestamp:.2f}s: {text[:50]}...")
                            break
    except Exception as e:
        logging.exception(f"Error in speech recognition: {e}")
    
    return speech_result, keyword_result

def parallel_speech_recognition(chunks, recognizer, trigger_patterns=None, use_ai=False):
    """
    Run speech recognition in parallel.
    
    Args:
        chunks: List of (timestamp, audio_chunk) tuples
        recognizer: SpeechRecognition recognizer instance
        trigger_patterns: Optional list of regex patterns to match in speech
        use_ai: Whether to use AI-based recognition as fallback
        
    Returns:
        Tuple containing (speech_results, keyword_results)
    """
    start_time = time.time()
    speech_results = []
    keyword_results = []
    total_chunks = len(chunks)
    
    logging.info(f"Processing {total_chunks} speech chunks in parallel")
    
    # Prepare task data
    task_data = [
        (i, timestamp, chunk, total_chunks, recognizer, trigger_patterns, use_ai)
        for i, (timestamp, chunk) in enumerate(chunks)
    ]
    
    # Determine optimal number of workers based on CPU cores
    max_workers = min(os.cpu_count() or 4, len(chunks))
    
    # Process speech chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_speech_chunk, data): i
            for i, data in enumerate(task_data)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                speech_result, keyword_result = future.result()
                
                # Store results if not None
                if speech_result:
                    speech_results.append(speech_result)
                
                if keyword_result:
                    keyword_results.append(keyword_result)
                    
            except Exception as e:
                logging.exception(f"Error processing chunk {chunk_idx}: {e}")
    
    # Sort results by timestamp
    speech_results.sort(key=lambda x: x[0])
    keyword_results.sort(key=lambda x: x[0])
    
    processing_time = time.time() - start_time
    logging.info(f"Parallel speech processing completed in {processing_time:.2f}s")
    logging.info(f"Found {len(speech_results)} speech segments and {len(keyword_results)} keyword matches")
    
    return speech_results, keyword_results
"""
Module for parallel speech processing in the screenshot extractor application.

This module provides functions to process audio chunks in parallel, improving
the performance of speech recognition in the video analysis pipeline.
"""

import os
import tempfile
import concurrent.futures
import wave
import io
import re
import time
import speech_recognition as sr
from typing import List, Tuple, Dict, Any, Optional

# Import required OpenAI utilities if available
try:
    from ...utils.openai_analyzer import OPENAI_AVAILABLE
    from ...utils.media_utils import transcribe_with_whisper
except ImportError:
    OPENAI_AVAILABLE = False

def process_audio_chunk(chunk_data):
    """
    Process a single audio chunk for speech recognition.
    
    Args:
        chunk_data: Tuple containing (chunk_index, chunk_time, chunk, chunk_count, 
                    recognizer, trigger_keywords, use_ai_speech_analysis)
    
    Returns:
        Tuple containing (speech_result, keyword_result)
    """
    # Unpack the chunk data
    chunk_index, chunk_time, chunk, chunk_count, recognizer, trigger_keywords, use_ai_speech_analysis = chunk_data
    
    # Convert chunk to format needed by SpeechRecognition
    chunk_file = io.BytesIO()
    chunk.export(chunk_file, format="wav")
    chunk_file.seek(0)
    
    speech_result = None
    keyword_result = None
    
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
                from ..audio.azure_speech_client import get_azure_speech_client
                azure_speech_client = get_azure_speech_client()
                if azure_speech_client.is_available():
                    result = azure_speech_client.transcribe_audio_segment(audio_segment)
                    text = result.get('text', '').strip()
                    print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with Azure AI Speech")
                else:
                    raise Exception("Azure AI Speech not available")
                    
            except Exception as azure_speech_err:
                print(f"Azure AI Speech failed for chunk {chunk_index+1}: {azure_speech_err}")
                
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
                        print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with {azure_client.service_type} Whisper")
                    else:
                        raise Exception("Azure OpenAI Whisper not available")
                        
                except Exception as whisper_service_err:
                    print(f"Azure OpenAI Whisper failed for chunk {chunk_index+1}: {whisper_service_err}")
                
                # Fallback to Google Speech Recognition
                try:
                    text = recognizer.recognize_google(audio_data)
                    print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with Google (fallback)")
                except Exception as google_err:
                    print(f"Google Speech Recognition failed: {google_err}")
                    
                    # Final fallback to OpenAI Whisper API if available
                    if OPENAI_AVAILABLE and use_ai_speech_analysis:
                        try:
                            # Export chunk to a temporary file for Whisper API
                            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                            temp_file.close()
                            
                            # Save audio data
                            with wave.open(temp_file.name, 'wb') as wf:
                                wf.setnchannels(1)  # Mono
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(16000)  # 16kHz
                                wf.writeframes(audio_data.get_raw_data())
                            
                            # Call OpenAI Whisper API
                            text = transcribe_with_whisper(temp_file.name)
                            print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with Whisper API (final fallback)")
                            
                            # Clean up temp file
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                        except Exception as whisper_api_err:
                            print(f"All speech recognition methods failed for chunk {chunk_index+1}: {whisper_api_err}")
            
            # Store transcript with timestamp
            if text:
                speech_result = (chunk_time, text)
                
                # Check for keywords in the transcript
                if trigger_keywords:
                    for pattern in trigger_keywords:
                        if re.search(pattern, text.lower()):
                            print(f"Keyword found at {chunk_time:.2f}s: {text[:120]}")
                            keyword_result = (chunk_time, f"Keyword trigger: {pattern}")
                            break
    except Exception as e:
        print(f"Error in speech recognition for chunk {chunk_index+1}: {e}")
        
    return speech_result, keyword_result

def run_parallel_speech_processing(audio_chunks, recognizer, trigger_keywords=None, use_ai_speech_analysis=False):
    """
    Process audio chunks in parallel to extract speech and keywords.
    
    Args:
        audio_chunks: List of (timestamp, audio_chunk) tuples
        recognizer: SpeechRecognition recognizer instance
        trigger_keywords: List of keyword patterns to search for
        use_ai_speech_analysis: Whether to use AI for speech analysis
        
    Returns:
        Tuple containing (speech_timestamps, keyword_timestamps)
    """
    if not audio_chunks:
        return [], []
    
    start_time = time.time()
    chunk_count = len(audio_chunks)
    
    # Determine optimal number of workers
    max_workers = min(os.cpu_count() or 4, chunk_count, 8)  # Limit to avoid API rate limits
    print(f"Running parallel speech processing with {max_workers} workers for {chunk_count} chunks")
    
    # Prepare data for parallel processing
    task_data = [
        (i, chunk_time, chunk, chunk_count, recognizer, trigger_keywords, use_ai_speech_analysis)
        for i, (chunk_time, chunk) in enumerate(audio_chunks)
    ]
    
    # Process chunks in parallel
    speech_timestamps = []
    keyword_timestamps = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_audio_chunk, data) for data in task_data]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                speech_result, keyword_result = future.result()
                
                if speech_result:
                    speech_timestamps.append(speech_result)
                    
                if keyword_result:
                    keyword_timestamps.append(keyword_result)
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
    
    # Sort results by timestamp
    speech_timestamps.sort(key=lambda x: x[0])
    keyword_timestamps.sort(key=lambda x: x[0])
    
    processing_time = time.time() - start_time
    print(f"Parallel speech processing completed in {processing_time:.2f}s")
    print(f"Found {len(speech_timestamps)} speech segments and {len(keyword_timestamps)} keywords")
    
    return speech_timestamps, keyword_timestamps
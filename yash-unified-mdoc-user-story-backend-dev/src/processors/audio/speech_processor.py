"""
Speech processor module for parallel speech recognition
This module provides functions for processing speech data in parallel
"""

import os
import tempfile
import concurrent.futures
import wave
import io
import re
import time
import speech_recognition as sr
from typing import List, Tuple, Any, Optional

# Check if OpenAI is available
try:
    from ...utils.openai_analyzer import OPENAI_AVAILABLE
    from ...utils.media_utils import transcribe_with_whisper
except ImportError:
    OPENAI_AVAILABLE = False

class SpeechProcessor:
    """Class for processing speech data in parallel"""
    
    def __init__(self, recognizer, trigger_keywords=None, use_ai_speech_analysis=False):
        """
        Initialize the speech processor
        
        Args:
            recognizer: SpeechRecognition recognizer instance
            trigger_keywords: List of keyword patterns to search for in speech
            use_ai_speech_analysis: Whether to use AI for speech analysis
        """
        self.recognizer = recognizer
        self.trigger_keywords = trigger_keywords or []
        self.use_ai_speech_analysis = use_ai_speech_analysis
        
    def process_chunk(self, chunk_data):
        """
        Process a single audio chunk
        
        Args:
            chunk_data: Tuple of (chunk_index, chunk_time, chunk)
            
        Returns:
            Tuple of (chunk_index, speech_result, keyword_result)
        """
        chunk_index, chunk_time, chunk = chunk_data
        chunk_count = chunk_data[3] if len(chunk_data) > 3 else 0
        
        # Convert chunk to format needed by SpeechRecognition
        chunk_file = io.BytesIO()
        chunk.export(chunk_file, format="wav")
        chunk_file.seek(0)
        
        speech_result = None
        keyword_result = None
        
        try:
            # Use speech recognition to convert audio to text
            with sr.AudioFile(chunk_file) as source:
                audio_data = self.recognizer.record(source)
                
                # Try with speech recognition services - Whisper first for best accuracy
                text = ""
                
                # First try local Whisper for superior accuracy
                try:
                    from .whisper_processor import get_optimized_whisper_processor
                    
                    # Save audio to temporary file for Whisper processing
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.close()
                    
                    with wave.open(temp_file.name, 'wb') as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)  # 16kHz
                        wf.writeframes(audio_data.get_raw_data())
                    
                    # Use local Whisper processor for superior accuracy
                    whisper_processor = get_optimized_whisper_processor()
                    result = whisper_processor.transcribe_audio(temp_file.name)
                    text = result.get('text', '').strip()
                    
                    print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with local Whisper")
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                        
                except Exception as whisper_local_err:
                    # First try Azure OpenAI Whisper for superior accuracy
                    try:
                        from .azure_whisper_client import get_azure_whisper_client
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
                        print(f"Whisper service failed for chunk {chunk_index+1}: {whisper_service_err}")
                        
                        # Fallback to Google Speech Recognition
                        try:
                            text = self.recognizer.recognize_google(audio_data)
                            print(f"Chunk {chunk_index+1}/{chunk_count}: Recognized speech with Google (fallback)")
                        except Exception as google_err:
                            print(f"Google Speech Recognition failed for chunk {chunk_index+1}: {google_err}")
                            
                            # Final fallback to OpenAI Whisper API if available
                            if OPENAI_AVAILABLE and self.use_ai_speech_analysis:
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
                                except:
                                    pass
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
                    if self.trigger_keywords:
                        for pattern in self.trigger_keywords:
                            if re.search(pattern, text.lower()):
                                print(f"Keyword found at {chunk_time:.2f}s: {text[:120]}")
                                keyword_result = (chunk_time, f"Keyword trigger: {pattern}")
                                break
        except Exception as e:
            print(f"Error in speech recognition for chunk {chunk_index+1}: {e}")
            
        return chunk_index, speech_result, keyword_result

def process_audio_chunks_parallel(audio_chunks, recognizer, trigger_keywords=None, 
                                use_ai_speech_analysis=False, max_workers=None):
    """
    Process audio chunks in parallel
    
    Args:
        audio_chunks: List of (chunk_time, chunk) tuples
        recognizer: SpeechRecognition recognizer instance
        trigger_keywords: List of keyword patterns to search for
        use_ai_speech_analysis: Whether to use AI for speech analysis
        max_workers: Maximum number of worker threads
        
    Returns:
        Tuple of (speech_timestamps, keyword_timestamps)
    """
    if not audio_chunks:
        return [], []
        
    # Create speech processor
    processor = SpeechProcessor(recognizer, trigger_keywords, use_ai_speech_analysis)
    
    # Determine number of workers (limited to avoid API rate limits)
    max_workers = max_workers or min(os.cpu_count() or 4, 8)
    
    # Print information about parallel processing
    start_time = time.time()
    chunk_count = len(audio_chunks)
    print(f"Processing {chunk_count} audio chunks in parallel with {max_workers} workers...")
    
    # Create task data with chunk indices and total count
    task_data = [(i, chunk_time, chunk, chunk_count) for i, (chunk_time, chunk) in enumerate(audio_chunks)]
    
    # Process chunks in parallel
    speech_timestamps = []
    keyword_timestamps = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunk processing tasks
        futures = [executor.submit(processor.process_chunk, data) for data in task_data]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_index, speech_result, keyword_result = future.result()
                
                # Add speech result if not None
                if speech_result:
                    speech_timestamps.append(speech_result)
                    
                # Add keyword result if not None
                if keyword_result:
                    keyword_timestamps.append(keyword_result)
            except Exception as e:
                print(f"Error processing speech chunk: {e}")
    
    # Sort results by timestamp
    speech_timestamps.sort(key=lambda x: x[0])
    keyword_timestamps.sort(key=lambda x: x[0])
    
    total_time = time.time() - start_time
    print(f"Parallel speech processing completed in {total_time:.2f}s")
    print(f"  - {len(speech_timestamps)} transcripts processed")
    print(f"  - {len(keyword_timestamps)} keywords found")
    
    return speech_timestamps, keyword_timestamps
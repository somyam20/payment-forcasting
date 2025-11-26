"""
Parallel Speech Processing Module

This module provides functionality for parallel processing of speech data
to extract keywords and transcript information from video audio.
"""

import os
import time
import concurrent.futures
import io
import tempfile
import wave
import re
from typing import List, Tuple, Dict, Any, Optional

class ParallelSpeechProcessor:
    """Helper class for parallel processing of speech data"""
    
    def __init__(self, extractor):
        """
        Initialize with reference to the parent extractor
        
        Args:
            extractor: Reference to the ScreenshotExtractor instance
        """
        self.extractor = extractor
        self.max_workers = max(2, min(os.cpu_count() or 4, 8))  # Limit to avoid API rate limits
        
    def process_audio_chunks_parallel(self, progress_callback=None):
        """
        Process audio chunks in parallel to extract speech and keywords
        
        Args:
            progress_callback: Optional progress callback
            
        Returns:
            Tuple of (speech_timestamps, keyword_timestamps)
        """
        if not self.extractor.audio_chunks:
            print("No audio chunks available for processing")
            return [], []
            
        start_time = time.time()
        print(f"Starting parallel speech processing with {len(self.extractor.audio_chunks)} chunks")
        
        # Create batches for better parallelization
        batch_size = max(1, min(5, len(self.extractor.audio_chunks) // self.max_workers))
        batches = []
        
        for i in range(0, len(self.extractor.audio_chunks), batch_size):
            batch = self.extractor.audio_chunks[i:i+batch_size]
            if batch:  # Only add non-empty batches
                batches.append((i // batch_size, batch))
                
        print(f"Split {len(self.extractor.audio_chunks)} chunks into {len(batches)} batches")
        
        # Process batches in parallel
        results = []
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_audio_batch, batch_id, chunks): batch_id 
                for batch_id, chunks in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_pct = completed / len(batches)
                        progress_callback(progress_pct, f"Processed audio batch {completed}/{len(batches)}")
                except Exception as e:
                    print(f"Error processing audio batch {batch_id}: {e}")
                    
        # Organize results
        speech_timestamps = []
        keyword_timestamps = []
        
        for result_type, timestamp, content in results:
            if result_type == "transcript":
                speech_timestamps.append((timestamp, content))
            elif result_type == "keyword":
                keyword_timestamps.append((timestamp, content))
                
        total_time = time.time() - start_time
        print(f"Parallel speech processing complete in {total_time:.2f}s")
        print(f"  - {len(speech_timestamps)} speech segments")
        print(f"  - {len(keyword_timestamps)} keyword timestamps")
        
        return speech_timestamps, keyword_timestamps
        
    def _process_audio_batch(self, batch_id, audio_chunks):
        """
        Process a batch of audio chunks
        
        Args:
            batch_id: ID of the batch
            audio_chunks: List of (timestamp, chunk) tuples
            
        Returns:
            List of (result_type, timestamp, content) tuples
        """
        batch_results = []
        start_time = time.time()
        
        for chunk_time, chunk in audio_chunks:
            # Skip empty chunks
            if len(chunk) == 0:
                continue
                
            # Convert to format needed by SpeechRecognition
            chunk_file = io.BytesIO()
            chunk.export(chunk_file, format="wav")
            chunk_file.seek(0)
            
            try:
                # Use speech recognition
                with self.extractor.recognizer.AudioFile(chunk_file) as source:
                    audio_data = self.extractor.recognizer.record(source)
                    
                    # Try with different APIs - Whisper first for best accuracy
                    text = ""
                    
                    # First try local Whisper for superior accuracy
                    try:
                        from ..audio.whisper_processor import get_optimized_whisper_processor
                        
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
                        
                        # Clean up temp file
                        try:
                            os.unlink(temp_file.name)
                        except:
                            pass
                            
                    except Exception as whisper_local_err:
                        # First try Azure OpenAI Whisper for superior accuracy
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
                                print(f"Recognized speech with {azure_client.service_type} Whisper")
                            else:
                                raise Exception("Azure OpenAI Whisper not available")
                                
                        except Exception as whisper_service_err:
                            print(f"Whisper service failed: {whisper_service_err}")
                            
                            # Fallback to Google Speech Recognition
                            try:
                                text = self.extractor.recognizer.recognize_google(audio_data)
                            except Exception as google_err:
                                # Final fallback to OpenAI Whisper API if available
                                if hasattr(self.extractor, 'use_ai_speech_analysis') and self.extractor.use_ai_speech_analysis:
                                    try:
                                        # Export to temporary file
                                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                        temp_file.close()
                                        
                                        # Write audio data
                                        with wave.open(temp_file.name, 'wb') as wf:
                                            wf.setnchannels(1)  # Mono
                                            wf.setsampwidth(2)  # 16-bit
                                            wf.setframerate(16000)  # 16kHz
                                            wf.writeframes(audio_data.get_raw_data())
                                        
                                        # Use Whisper API
                                        from ...utils.media_utils import transcribe_with_whisper
                                        text = transcribe_with_whisper(temp_file.name)
                                        
                                        # Clean up
                                        try:
                                            os.unlink(temp_file.name)
                                        except:
                                            pass
                                    except Exception as whisper_api_err:
                                        print(f"All speech recognition methods failed in batch {batch_id}: {whisper_api_err}")
                        
                    # Store transcript for reference
                    if text:
                        print(f"Batch {batch_id} - Transcript at {chunk_time:.2f}s: {text[:50]}...")
                        batch_results.append(("transcript", chunk_time, text))
                        
                        # Check for keywords in the transcript
                        if hasattr(self.extractor, 'keyword_trigger') and self.extractor.keyword_trigger:
                            for pattern in self.extractor.trigger_keywords:
                                if re.search(pattern, text.lower()):
                                    print(f"Batch {batch_id} - Keyword at {chunk_time:.2f}s: {pattern}")
                                    batch_results.append(("keyword", chunk_time, f"Keyword trigger: {pattern}"))
                                    break
            except Exception as e:
                print(f"Error in batch {batch_id} at {chunk_time:.2f}s: {e}")
                
        processing_time = time.time() - start_time
        print(f"Batch {batch_id}: Processed {len(audio_chunks)} chunks in {processing_time:.2f}s")
        
        return batch_results
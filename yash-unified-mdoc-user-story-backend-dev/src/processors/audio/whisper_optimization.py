"""
Whisper Performance Optimization Module

This module provides advanced optimization techniques for Whisper transcription
in parallel processing scenarios, including batch processing, memory management,
and efficient audio preprocessing.
"""

import os
import tempfile
import threading
from typing import List, Tuple, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pydub import AudioSegment


class WhisperBatchProcessor:
    """
    Optimized batch processor for Whisper transcription with shared model instance
    """
    
    def __init__(self, whisper_processor, max_workers=None, batch_size=5):
        """
        Initialize batch processor
        
        Args:
            whisper_processor: Shared WhisperProcessor instance
            max_workers: Maximum number of worker threads
            batch_size: Number of audio chunks to process per batch
        """
        self.whisper_processor = whisper_processor
        self.max_workers = max_workers or min(4, os.cpu_count())
        self.batch_size = batch_size
        self.processing_lock = threading.Lock()
        
    def process_audio_chunks_optimized(self, audio_chunks: List[Tuple[float, AudioSegment]], 
                                     progress_callback=None) -> List[Tuple[float, str]]:
        """
        Process multiple audio chunks with optimized batching and shared model
        
        Args:
            audio_chunks: List of (timestamp, audio_segment) tuples
            progress_callback: Optional progress callback function
            
        Returns:
            List of (timestamp, transcription) tuples
        """
        if not audio_chunks:
            return []
        
        print(f"Processing {len(audio_chunks)} audio chunks with optimized Whisper batching")
        start_time = time.time()
        
        # Create batches for processing
        batches = self._create_batches(audio_chunks)
        results = []
        processed_count = 0
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._process_batch_optimized, batch_id, batch): batch_id 
                for batch_id, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    # Report progress
                    if progress_callback:
                        progress = min(100, (processed_count / len(audio_chunks)) * 100)
                        progress_callback(progress, f"Processed {processed_count}/{len(audio_chunks)} chunks")
                        
                except Exception as e:
                    print(f"Batch {batch_id} processing failed: {e}")
        
        # Sort results by timestamp
        results.sort(key=lambda x: x[0])
        
        elapsed_time = time.time() - start_time
        print(f"✓ Whisper batch processing complete: {len(results)} transcriptions in {elapsed_time:.2f}s")
        
        return results
    
    def _create_batches(self, audio_chunks: List[Tuple[float, AudioSegment]]) -> List[List[Tuple[float, AudioSegment]]]:
        """Create optimized batches from audio chunks"""
        batches = []
        for i in range(0, len(audio_chunks), self.batch_size):
            batch = audio_chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _process_batch_optimized(self, batch_id: int, batch: List[Tuple[float, AudioSegment]]) -> List[Tuple[float, str]]:
        """
        Process a single batch of audio chunks with optimized Whisper usage and robust error handling
        
        Args:
            batch_id: Batch identifier
            batch: List of (timestamp, audio_segment) tuples
            
        Returns:
            List of (timestamp, transcription) tuples
        """
        batch_results = []
        temp_files = []
        
        try:
            # Pre-process all audio in batch to temporary files with validation
            batch_audio_files = []
            for timestamp, audio_segment in batch:
                try:
                    # Skip empty or very short audio segments
                    if len(audio_segment) < 500:  # Less than 0.5 seconds
                        print(f"Batch {batch_id}: Skipping short audio at {timestamp:.2f}s ({len(audio_segment)}ms)")
                        continue
                    
                    # Validate audio has actual content (not silence)
                    if audio_segment.dBFS < -50:  # Very quiet audio
                        print(f"Batch {batch_id}: Skipping silent audio at {timestamp:.2f}s ({audio_segment.dBFS:.1f}dBFS)")
                        continue
                    
                    # Create temporary file for this chunk
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.close()
                    temp_files.append(temp_file.name)
                    
                    # Optimize audio format for Whisper processing
                    optimized_audio = audio_segment.set_channels(1).set_frame_rate(16000)
                    
                    # Export audio to optimal format for Whisper
                    optimized_audio.export(
                        temp_file.name,
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"]
                    )
                    
                    # Verify the exported file has content
                    if os.path.getsize(temp_file.name) > 1000:  # At least 1KB
                        batch_audio_files.append((timestamp, temp_file.name))
                    else:
                        print(f"Batch {batch_id}: Exported file too small at {timestamp:.2f}s")
                    
                except Exception as audio_prep_error:
                    print(f"Batch {batch_id}: Audio preprocessing failed at {timestamp:.2f}s - {audio_prep_error}")
                    continue
            
            # Process all valid files in batch using shared model (with thread safety)
            with self.processing_lock:
                for timestamp, audio_file in batch_audio_files:
                    try:
                        # Use shared Whisper processor for transcription
                        result = self.whisper_processor.transcribe_audio(audio_file)
                        text = result.get('text', '').strip()
                        
                        if text and len(text) > 3:  # Minimum meaningful text length
                            batch_results.append((timestamp, text))
                            print(f"Batch {batch_id}: Transcribed {timestamp:.2f}s - {text[:50]}...")
                        else:
                            print(f"Batch {batch_id}: No meaningful text at {timestamp:.2f}s")
                        
                    except Exception as e:
                        print(f"Batch {batch_id}: Transcription failed at {timestamp:.2f}s - {e}")
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        print(f"Batch {batch_id}: Completed with {len(batch_results)} successful transcriptions")
        return batch_results


class WhisperMemoryOptimizer:
    """
    Memory optimization utilities for Whisper processing
    """
    
    @staticmethod
    def optimize_audio_for_whisper(audio_segment: AudioSegment) -> AudioSegment:
        """
        Optimize audio segment for Whisper processing
        
        Args:
            audio_segment: Input audio segment
            
        Returns:
            Optimized audio segment
        """
        # Convert to optimal format for Whisper
        optimized = audio_segment.set_channels(1)      # Mono
        optimized = optimized.set_frame_rate(16000)    # 16kHz sample rate
        optimized = optimized.set_sample_width(2)      # 16-bit depth
        
        # Normalize audio levels
        optimized = optimized.normalize()
        
        return optimized
    
    @staticmethod
    def split_long_audio(audio_segment: AudioSegment, max_duration_seconds=30) -> List[AudioSegment]:
        """
        Split long audio into smaller chunks for better Whisper performance
        
        Args:
            audio_segment: Input audio segment
            max_duration_seconds: Maximum duration per chunk
            
        Returns:
            List of audio chunks
        """
        if len(audio_segment) <= max_duration_seconds * 1000:
            return [audio_segment]
        
        chunks = []
        chunk_length_ms = max_duration_seconds * 1000
        
        for i in range(0, len(audio_segment), chunk_length_ms):
            chunk = audio_segment[i:i + chunk_length_ms]
            chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def preprocess_audio_batch(audio_chunks: List[Tuple[float, AudioSegment]]) -> List[Tuple[float, AudioSegment]]:
        """
        Preprocess a batch of audio chunks for optimal Whisper performance
        
        Args:
            audio_chunks: List of (timestamp, audio_segment) tuples
            
        Returns:
            List of optimized (timestamp, audio_segment) tuples
        """
        optimized_chunks = []
        
        for timestamp, audio_segment in audio_chunks:
            # Skip very short audio (less than 0.5 seconds)
            if len(audio_segment) < 500:
                continue
            
            # Optimize for Whisper
            optimized_audio = WhisperMemoryOptimizer.optimize_audio_for_whisper(audio_segment)
            
            # Split if too long
            if len(optimized_audio) > 30000:  # 30 seconds
                sub_chunks = WhisperMemoryOptimizer.split_long_audio(optimized_audio, 25)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_timestamp = timestamp + (i * 25)  # Offset by chunk duration
                    optimized_chunks.append((sub_timestamp, sub_chunk))
            else:
                optimized_chunks.append((timestamp, optimized_audio))
        
        return optimized_chunks


def create_optimized_whisper_pipeline(max_workers=None, batch_size=5):
    """
    Create an optimized Whisper processing pipeline
    
    Args:
        max_workers: Maximum number of worker threads
        batch_size: Batch size for processing
        
    Returns:
        Tuple of (batch_processor, memory_optimizer)
    """
    from .whisper_processor import get_optimized_whisper_processor
    
    # Get shared Whisper processor
    whisper_processor = get_optimized_whisper_processor()
    
    # Create optimized batch processor
    batch_processor = WhisperBatchProcessor(
        whisper_processor=whisper_processor,
        max_workers=max_workers,
        batch_size=batch_size
    )
    
    # Create memory optimizer
    memory_optimizer = WhisperMemoryOptimizer()
    
    return batch_processor, memory_optimizer


def transcribe_audio_chunks_optimized(audio_chunks: List[Tuple[float, AudioSegment]], 
                                    progress_callback=None) -> List[Tuple[float, str]]:
    """
    High-performance transcription of audio chunks using optimized Whisper pipeline
    
    Args:
        audio_chunks: List of (timestamp, audio_segment) tuples
        progress_callback: Optional progress callback
        
    Returns:
        List of (timestamp, transcription) tuples
    """
    if not audio_chunks:
        return []
    
    print(f"Starting optimized Whisper transcription for {len(audio_chunks)} chunks")
    
    # Create optimized pipeline
    batch_processor, memory_optimizer = create_optimized_whisper_pipeline()
    
    # Preprocess audio for optimal performance
    if progress_callback:
        progress_callback(5, "Preprocessing audio chunks...")
    
    optimized_chunks = memory_optimizer.preprocess_audio_batch(audio_chunks)
    print(f"Preprocessed {len(audio_chunks)} chunks into {len(optimized_chunks)} optimized chunks")
    
    if progress_callback:
        progress_callback(10, "Starting batch transcription...")
    
    # Process with optimized batch processor
    def batch_progress_callback(progress, message):
        if progress_callback:
            # Map batch progress to overall progress (10% to 95%)
            overall_progress = 10 + (progress * 0.85)
            progress_callback(overall_progress, message)
    
    results = batch_processor.process_audio_chunks_optimized(
        optimized_chunks, 
        progress_callback=batch_progress_callback
    )
    
    if progress_callback:
        progress_callback(100, f"Transcription complete: {len(results)} results")
    
    return results
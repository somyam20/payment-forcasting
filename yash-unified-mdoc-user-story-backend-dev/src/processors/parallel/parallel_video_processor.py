"""
Parallel Video Processing Module

This module enhances the screenshot extraction process by enabling parallel processing
of video chunks, speech recognition, and frame analysis to significantly improve
performance for larger videos.
"""

import os
import time
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
from PIL import Image
import io

class ParallelVideoProcessor:
    """Helper class for parallel processing of video analysis tasks"""
    
    def __init__(self, max_workers=None):
        """
        Initialize the parallel processor
        
        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count)
        """
        self.max_workers = max_workers or max(2, os.cpu_count() or 4)
        print(f"Parallel video processor initialized with {self.max_workers} workers")
    
    def process_speech_in_parallel(self, audio_chunks, process_chunk_func, progress_callback=None):
        """
        Process audio chunks in parallel for speech recognition
        
        Args:
            audio_chunks: List of (timestamp, audio_chunk) tuples
            process_chunk_func: Function to process each audio chunk
            progress_callback: Optional progress callback function
            
        Returns:
            List of combined results from all processed chunks
        """
        if not audio_chunks:
            return []
            
        start_time = time.time()
        print(f"Starting parallel speech processing with {len(audio_chunks)} chunks")
        
        # Split audio chunks into batches for better parallelization
        batch_size = max(1, len(audio_chunks) // self.max_workers)
        batches = []
        
        for i in range(0, len(audio_chunks), batch_size):
            batch = audio_chunks[i:i+batch_size]
            if batch:  # Only add non-empty batches
                batches.append((i // batch_size, batch))
        
        print(f"Split audio into {len(batches)} batches for parallel processing")
        
        # Process batches in parallel
        all_results = []
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._process_audio_batch, batch_id, chunks, process_chunk_func): batch_id 
                for batch_id, chunks in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_percent = completed / len(batches)
                        progress_callback(
                            progress_percent, 
                            f"Processed audio batch {completed}/{len(batches)}"
                        )
                except Exception as e:
                    print(f"Error processing audio batch {batch_id}: {e}")
        
        total_time = time.time() - start_time
        print(f"Parallel speech processing complete: {len(all_results)} results in {total_time:.2f}s")
        
        return all_results
    
    def _process_audio_batch(self, batch_id, audio_chunks, process_chunk_func):
        """Process a batch of audio chunks with the provided function"""
        batch_results = []
        start_time = time.time()
        
        for chunk_time, chunk in audio_chunks:
            # Process each chunk with the provided function
            chunk_results = process_chunk_func(chunk_time, chunk)
            if chunk_results:
                batch_results.extend(chunk_results)
        
        processing_time = time.time() - start_time
        print(f"Batch {batch_id}: Processed {len(audio_chunks)} audio chunks in {processing_time:.2f}s")
        
        return batch_results
    
    def process_video_timestamps_in_parallel(self, timestamps, process_timestamp_func, 
                                          video_processor, fps, frame_count, progress_callback=None):
        """
        Process video timestamps in parallel to extract screenshots
        
        Args:
            timestamps: List of (timestamp, reason) tuples to process
            process_timestamp_func: Function to process each timestamp
            video_processor: VideoProcessor instance for frame access
            fps: Frames per second
            frame_count: Total frame count
            progress_callback: Optional progress callback function
            
        Returns:
            List of screenshots from all processed timestamps
        """
        if not timestamps:
            return []
            
        start_time = time.time()
        print(f"Starting parallel timestamp processing with {len(timestamps)} timestamps")
        
        # Group timestamps into batches for parallel processing
        # Use fewer batches than max_workers to minimize thread contention
        optimal_batch_count = max(2, min(self.max_workers, len(timestamps) // 3))
        batch_size = max(1, len(timestamps) // optimal_batch_count)
        batches = []
        
        for i in range(0, len(timestamps), batch_size):
            batch = timestamps[i:i+batch_size]
            if batch:  # Only add non-empty batches
                batches.append((i // batch_size, batch))
        
        print(f"Split timestamps into {len(batches)} batches for parallel processing")
        
        # Process batches in parallel
        all_screenshots = []
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    self._process_timestamp_batch, 
                    batch_id, 
                    ts_batch, 
                    process_timestamp_func,
                    video_processor,
                    fps,
                    frame_count
                ): batch_id for batch_id, ts_batch in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_screenshots = future.result()
                    all_screenshots.extend(batch_screenshots)
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_percent = completed / len(batches)
                        progress_callback(
                            progress_percent, 
                            f"Processed timestamp batch {completed}/{len(batches)}"
                        )
                except Exception as e:
                    print(f"Error processing timestamp batch {batch_id}: {e}")
        
        # Sort screenshots by timestamp for consistency
        all_screenshots.sort(key=lambda x: x[1])
        
        total_time = time.time() - start_time
        print(f"Parallel timestamp processing complete: {len(all_screenshots)} screenshots in {total_time:.2f}s")
        
        return all_screenshots
    
    def _process_timestamp_batch(self, batch_id, timestamps, process_func, 
                               video_processor, fps, frame_count):
        """Process a batch of timestamps with the provided function"""
        batch_screenshots = []
        start_time = time.time()
        processed_frames = set()  # Local set to track processed frames in this batch
        
        for timestamp, reason in timestamps:
            # Process the timestamp with the provided function
            screenshot = process_func(
                timestamp, 
                reason, 
                video_processor, 
                fps, 
                frame_count, 
                processed_frames
            )
            
            if screenshot:
                batch_screenshots.extend(screenshot)
        
        processing_time = time.time() - start_time
        print(f"Batch {batch_id}: Processed {len(timestamps)} timestamps in {processing_time:.2f}s")
        
        return batch_screenshots
        
    def run_parallel_tasks(self, tasks):
        """
        Run multiple independent tasks in parallel
        
        Args:
            tasks: List of (func, args) tuples to execute in parallel
            
        Returns:
            List of results from all tasks
        """
        if not tasks:
            return []
            
        start_time = time.time()
        print(f"Running {len(tasks)} tasks in parallel")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(func, *args) for func, args in tasks]
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in parallel task: {e}")
        
        total_time = time.time() - start_time
        print(f"Parallel tasks complete in {total_time:.2f}s")
        
        return results
"""
Parallel Screenshot Extractor Module

This module enhances the screenshot extraction process with parallel processing capabilities.
It wraps the standard screenshot extractor and enables parallel processing of video chunks.
"""

import os
import time
import concurrent.futures
from typing import List, Tuple, Dict, Any, Optional
import math
from PIL import Image
import numpy as np

class ParallelExtractor:
    """
    Class for parallel processing of video chunks to extract screenshots
    """
    
    def __init__(self, screenshot_extractor, video_processor, fps, frame_count):
        """
        Initialize the parallel extractor
        
        Args:
            screenshot_extractor: The ScreenshotExtractor instance to use
            video_processor: VideoProcessor instance for frame access
            fps: Frames per second
            frame_count: Total frame count
        """
        self.extractor = screenshot_extractor
        self.video_processor = video_processor
        self.fps = fps
        self.frame_count = frame_count
        self.max_workers = os.cpu_count() or 4
        
    def process_in_parallel(self, progress_callback=None):
        """
        Process the video in parallel by:
        1. Running speech analysis and scene detection concurrently
        2. Processing key timestamps in parallel chunks
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            list: List of screenshots
        """
        start_time = time.time()
        print(f"Starting parallel video processing with {self.max_workers} workers")
        
        # Phase 1: Run speech processing and scene detection in parallel
        if progress_callback:
            progress_callback(0.0, "Prosessing...")
        
        phase1_results = self._run_parallel_phase1(progress_callback)
        
        # Extract key timestamps from Phase 1
        all_timestamps = self._combine_phase1_results(phase1_results)
        
        if progress_callback:
            progress_callback(0.5, f"Processing...")
            
        # Phase 2: Process key timestamps in parallel
        screenshots = self._process_timestamps_parallel(all_timestamps, progress_callback)
        
        total_time = time.time() - start_time
        print(f"Parallel processing complete: {len(screenshots)} screenshots in {total_time:.2f}s")
        
        return screenshots
        
    def _run_parallel_phase1(self, progress_callback=None):
        """
        Run Phase 1 tasks (speech processing and scene detection) in parallel
        
        Returns:
            Dict containing speech_results and scene_results
        """
        # Create tasks to run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit speech processing task
            speech_future = executor.submit(self.extractor.extract_all_speech_keywords)
            
            # Submit scene detection task
            scene_future = executor.submit(
                self.extractor.detect_scene_changes_fast,
                self.video_processor,
                self.fps,
                self.frame_count,
                lambda p, m: progress_callback(0.2 + p * 0.25, m) if progress_callback else None
            )
            
            # Wait for both tasks to complete
            speech_results = speech_future.result()
            scene_results = scene_future.result()
            
        return {
            "speech_results": speech_results,
            "scene_results": scene_results
        }
        
    def _combine_phase1_results(self, phase1_results):
        """
        Combine results from Phase 1 and filter to get key timestamps
        
        Returns:
            List of (timestamp, reason) tuples
        """
        # Combine all timestamps and sort
        all_timestamps = []
        
        # Add keyword timestamps
        all_timestamps.extend(self.extractor.keyword_timestamps)
        
        # Add scene change timestamps
        if len(self.extractor.scene_change_timestamps) > 0:
            print(f"Adding {len(self.extractor.scene_change_timestamps)} scene changes to process")
            all_timestamps.extend(self.extractor.scene_change_timestamps)
        else:
            print("Warning: No scene changes detected or enabled")
        
        # Add AI-detected timestamps if they exist
        if self.extractor.use_ai_speech_analysis:
            all_timestamps.extend(self.extractor.ai_timestamps)
            
        # Add start and end frames
        all_timestamps.append((0, "Start of video"))
        all_timestamps.append((self.frame_count / self.fps, "End of video"))
        
        # Print all collected timestamps for debugging
        print(f"All collected timestamps before filtering: {len(all_timestamps)}")
        
        # Sort by timestamp and remove duplicates or too close timestamps
        all_timestamps.sort(key=lambda x: x[0])
        
        filtered_timestamps = []
        last_time = -self.extractor.cooldown
        
        for timestamp, reason in all_timestamps:
            # Only keep timestamps that are at least cooldown seconds apart
            if timestamp - last_time >= self.extractor.cooldown:
                filtered_timestamps.append((timestamp, reason))
                last_time = timestamp
                
        # Print filtered timestamps for debugging
        print(f"After filtering: {len(filtered_timestamps)} timestamps to process")
        
        # Store the final list of timestamps to process
        self.extractor.key_timestamps = filtered_timestamps
        
        return filtered_timestamps
        
    def _process_timestamps_parallel(self, key_timestamps, progress_callback=None):
        """
        Process key timestamps in parallel to extract screenshots
        
        Args:
            key_timestamps: List of (timestamp, reason) tuples
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of screenshots
        """
        # Split timestamps into batches for parallel processing
        batch_count = min(self.max_workers, max(2, len(key_timestamps) // 5))
        batch_size = max(1, len(key_timestamps) // batch_count)
        batches = []
        
        for i in range(0, len(key_timestamps), batch_size):
            batch = key_timestamps[i:i+batch_size]
            if batch:  # Only add non-empty batches
                batches.append((i // batch_size, batch))
                
        print(f"Split {len(key_timestamps)} timestamps into {len(batches)} batches")
        
        # Process batches in parallel
        screenshots = []
        completed = 0
        shared_processed_frames = set()  # Track processed frames globally
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(
                    self._process_timestamp_batch,
                    batch_id,
                    batch_timestamps,
                    shared_processed_frames
                ): batch_id for batch_id, batch_timestamps in batches
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_screenshots = future.result()
                    screenshots.extend(batch_screenshots)
                    
                    # Update progress
                    completed += 1
                    if progress_callback:
                        progress_percent = 0.5 + (completed / len(batches)) * 0.5
                        progress_callback(
                            progress_percent, 
                            f"Processed batch {completed}/{len(batches)}"
                        )
                except Exception as e:
                    print(f"Error processing batch {batch_id}: {e}")
        
        # Sort screenshots by timestamp for consistency
        screenshots.sort(key=lambda x: x[1])
        
        return screenshots
        
    def _process_timestamp_batch(self, batch_id, timestamps, shared_processed_frames):
        """
        Process a batch of timestamps to extract screenshots
        
        Args:
            batch_id: ID of the batch
            timestamps: List of (timestamp, reason) tuples
            shared_processed_frames: Set to track globally processed frames
            
        Returns:
            List of (image, timestamp, reason) screenshots
        """
        batch_screenshots = []
        start_time = time.time()
        # Local set to track processed frames in this batch
        local_processed_frames = set()
        local_prev_frame = None
        
        for timestamp, reason in timestamps:
            # Convert timestamp to frame number
            center_frame_number = int(timestamp * self.fps)
            
            # Skip if this frame has already been processed globally
            if center_frame_number in shared_processed_frames:
                continue
                
            # Track that we've processed this frame globally
            shared_processed_frames.add(center_frame_number)
                
            # Special handling for scene changes - use exact frame
            if "Scene change" in reason:
                frame = self.video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    batch_screenshots.append((img_pil, timestamp, reason))
                    local_processed_frames.add(center_frame_number)
                continue  # Skip window processing for scene changes
            
            # Special handling for AI detection - always capture exact frame
            elif "AI detected" in reason or "AI Detected" in reason:
                frame = self.video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    batch_screenshots.append((img_pil, timestamp, reason))
                    local_processed_frames.add(center_frame_number)
                
            # Process a window around the timestamp to find the best frame
            window_size = 10  # frames to check on each side
            best_frame = None
            best_reason = reason
            
            # Check frames in the window
            for offset in range(-window_size, window_size + 1):
                frame_number = center_frame_number + offset
                
                # Skip if out of range
                if frame_number < 0 or frame_number >= self.frame_count:
                    continue
                    
                # Skip if already processed (locally or globally)
                if frame_number in local_processed_frames or frame_number in shared_processed_frames:
                    continue
                    
                # Mark as processed
                local_processed_frames.add(frame_number)
                shared_processed_frames.add(frame_number)
                
                # Get the frame
                frame = self.video_processor.get_frame_at_position(frame_number)
                if frame is None:
                    continue
                    
                # Calculate exact timestamp
                exact_timestamp = frame_number / self.fps
                
                # Run key frame detection
                if local_prev_frame is not None:
                    is_key, detection_reason = self.extractor.is_key_frame(frame, exact_timestamp)
                    
                    if is_key:
                        # Convert frame to PIL Image
                        img_pil = Image.fromarray(frame)
                        
                        # Use detection reason if it's more specific
                        if detection_reason:
                            best_reason = detection_reason
                            
                        # Add to screenshots
                        batch_screenshots.append((img_pil, exact_timestamp, best_reason))
                        break  # Found a good frame, move to next timestamp
                
                # Store this frame as previous for next iteration
                local_prev_frame = frame
                
        processing_time = time.time() - start_time
        print(f"Batch {batch_id}: Processed {len(timestamps)} timestamps in {processing_time:.2f}s")
        
        return batch_screenshots
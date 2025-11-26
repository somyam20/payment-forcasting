"""
Parallel video chunk processing module.

This module provides functions to process video chunks in parallel,
significantly speeding up screenshot extraction and document generation.
"""

import concurrent.futures
import os
import time
import math
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Any, Optional
import io

class VideoChunkProcessor:
    """
    Processes video chunks in parallel for optimal performance.
    """
    
    def __init__(self, extractor, video_processor, fps, frame_count):
        """
        Initialize the chunk processor.
        
        Args:
            extractor: ScreenshotExtractor instance to use for processing
            video_processor: VideoProcessor for accessing frames
            fps: Frames per second of the video
            frame_count: Total frame count
        """
        self.extractor = extractor
        self.video_processor = video_processor
        self.fps = fps
        self.frame_count = frame_count
        self.max_workers = os.cpu_count() or 4
        
    def split_video_into_chunks(self, chunk_duration_seconds=30):
        """
        Split the video into chunks for parallel processing.
        
        Args:
            chunk_duration_seconds: Duration of each chunk in seconds
            
        Returns:
            List of chunk data (start_frame, end_frame, start_time, end_time)
        """
        # Calculate chunk size in frames
        chunk_size_frames = int(chunk_duration_seconds * self.fps)
        
        # Create chunks
        chunks = []
        for start_frame in range(0, self.frame_count, chunk_size_frames):
            end_frame = min(start_frame + chunk_size_frames, self.frame_count)
            start_time = start_frame / self.fps
            end_time = end_frame / self.fps
            chunks.append((start_frame, end_frame, start_time, end_time))
        
        print(f"Split video into {len(chunks)} chunks of {chunk_duration_seconds}s each")
        return chunks
    
    def process_chunk(self, chunk_data):
        """
        Process a single video chunk.
        
        Args:
            chunk_data: Tuple of (chunk_id, start_frame, end_frame, start_time, end_time)
            
        Returns:
            Tuple of (chunk_id, screenshots, timestamps)
        """
        chunk_id, start_frame, end_frame, start_time, end_time = chunk_data
        chunk_screenshots = []
        
        print(f"Processing chunk {chunk_id}: frames {start_frame}-{end_frame}, time {start_time:.2f}s-{end_time:.2f}s")
        start_processing_time = time.time()
        
        # Initialize local state for this chunk
        prev_frame = None
        prev_text = None
        prev_gray = None
        frame_buffer = []
        processed_frames = set()
        
        # Get key timestamps that fall within this chunk's time range
        chunk_timestamps = []
        try:
            # Make sure key_timestamps exists and is not empty
            if hasattr(self.extractor, 'key_timestamps') and self.extractor.key_timestamps:
                for timestamp, reason in self.extractor.key_timestamps:
                    if start_time <= timestamp <= end_time:
                        chunk_timestamps.append((timestamp, reason))
        except Exception as e:
            print(f"Error getting timestamps for chunk {chunk_id}: {e}")
            # Create default timestamps in this chunk's range
            num_timestamps = max(2, int((end_time - start_time) / 5))  # One every 5 seconds
            for i in range(num_timestamps):
                ts = start_time + (i * (end_time - start_time)) / (num_timestamps - 1) if num_timestamps > 1 else start_time
                chunk_timestamps.append((ts, f"Default timestamp in chunk {chunk_id}"))
        
        # If we have specific timestamps in this chunk
        if chunk_timestamps:
            print(f"Chunk {chunk_id}: Processing {len(chunk_timestamps)} key timestamps")
            
            # Process each key timestamp
            for timestamp, reason in chunk_timestamps:
                # Convert timestamp to frame number
                center_frame = int(timestamp * self.fps)
                
                # Process a small window around the timestamp
                window_size = 5  # frames to check on each side
                best_screenshot = None
                
                for offset in range(-window_size, window_size + 1):
                    frame_number = center_frame + offset
                    
                    # Skip if out of range or already processed
                    if frame_number < start_frame or frame_number >= end_frame or frame_number in processed_frames:
                        continue
                    
                    # Get the frame
                    frame = self.video_processor.get_frame_at_position(frame_number)
                    if frame is None:
                        continue
                    
                    # Mark as processed
                    processed_frames.add(frame_number)
                    
                    # Calculate exact timestamp
                    exact_timestamp = frame_number / self.fps
                    
                    # For scene changes, always use the exact frame
                    if "Scene change" in reason:
                        print(f"Chunk {chunk_id}: Processing scene change at {exact_timestamp:.2f}s")
                        img_pil = Image.fromarray(frame)
                        chunk_screenshots.append((img_pil, exact_timestamp, reason))
                        break
                    
                    # For AI detection, always capture the exact frame
                    elif "AI detected" in reason or "AI Detected" in reason:
                        print(f"Chunk {chunk_id}: Processing AI detection at {exact_timestamp:.2f}s: {reason}")
                        img_pil = Image.fromarray(frame)
                        chunk_screenshots.append((img_pil, exact_timestamp, reason))
                        # Continue processing to see if we can find a better frame
                        
                    # For keyword detection, print detailed information
                    elif "Keyword" in reason:
                        print(f"Chunk {chunk_id}: Processing keyword detection at {exact_timestamp:.2f}s: {reason}")
                        # Continue to process with UI change detection
                    
                    # For other timestamps (like keywords), add to buffer and check key frame logic
                    if len(frame_buffer) >= 15:
                        frame_buffer.pop(0)
                    frame_buffer.append((frame, exact_timestamp))
                    
                    if prev_frame is not None:
                        # Temporarily set extractor state
                        original_prev_frame = self.extractor.prev_frame
                        original_frame_buffer = self.extractor.frame_buffer
                        
                        # Set temp state
                        self.extractor.prev_frame = prev_frame
                        self.extractor.frame_buffer = frame_buffer
                        
                        # Check if this is a key frame using SSIM and other detection methods
                        is_key, detection_reason = self.extractor.is_key_frame(frame, exact_timestamp)
                        
                        # Restore original state
                        self.extractor.prev_frame = original_prev_frame
                        self.extractor.frame_buffer = original_frame_buffer
                        
                        # If SSIM detected a change, log it
                        if is_key and "SSIM" in detection_reason:
                            print(f"Chunk {chunk_id}: UI change detected at {exact_timestamp:.2f}s with SSIM: {detection_reason}")
                        # If text change detected, log it
                        elif is_key and "text change" in detection_reason.lower():
                            print(f"Chunk {chunk_id}: Text change detected at {exact_timestamp:.2f}s: {detection_reason}")
                        # If mouse click detected, log it
                        elif is_key and "click" in detection_reason.lower():
                            print(f"Chunk {chunk_id}: Mouse click detected at {exact_timestamp:.2f}s: {detection_reason}")
                            
                        if is_key:
                            img_pil = Image.fromarray(frame)
                            # Use the detection reason if it's more specific
                            final_reason = detection_reason if detection_reason else reason
                            best_screenshot = (img_pil, exact_timestamp, final_reason)
                            print(f"Chunk {chunk_id}: Found key frame at {exact_timestamp:.2f}s - {final_reason}")
                            # Found a good frame, no need to check more
                            break
                    
                    # Update prev frame
                    prev_frame = frame
                
                # Add the best screenshot we found (if any)
                if best_screenshot:
                    chunk_screenshots.append(best_screenshot)
                    print(f"Chunk {chunk_id}: Screenshot at {best_screenshot[1]:.2f}s - {best_screenshot[2]}")
        
        # If no specific timestamps or we want additional frames, scan the entire chunk
        if not chunk_timestamps or len(chunk_screenshots) == 0:
            print(f"Chunk {chunk_id}: Scanning entire chunk for changes")
            
            # Reset for full scan
            prev_frame = None
            frame_buffer = []
            
            # Process frames with reasonable sampling (every 15th frame)
            for frame_number in range(start_frame, end_frame, 15):
                if frame_number in processed_frames:
                    continue
                    
                # Get the frame
                frame = self.video_processor.get_frame_at_position(frame_number)
                if frame is None:
                    continue
                    
                # Calculate timestamp
                timestamp = frame_number / self.fps
                
                # Add to frame buffer
                if len(frame_buffer) >= 15:
                    frame_buffer.pop(0)
                frame_buffer.append((frame, timestamp))
                
                # Run key frame detection
                if prev_frame is not None:
                    # Temporarily set extractor state
                    original_prev_frame = self.extractor.prev_frame
                    original_frame_buffer = self.extractor.frame_buffer
                    
                    # Set temp state
                    self.extractor.prev_frame = prev_frame
                    self.extractor.frame_buffer = frame_buffer
                    
                    # Run structural similarity detection and other change detection methods
                    is_key, reason = self.extractor.is_key_frame(frame, timestamp)
                    
                    # Restore original state
                    self.extractor.prev_frame = original_prev_frame
                    self.extractor.frame_buffer = original_frame_buffer
                    
                    # Detailed logging based on detection type
                    if is_key:
                        # Log detection type
                        if "SSIM" in reason:
                            print(f"Chunk {chunk_id}: Full scan - UI change detected at {timestamp:.2f}s with SSIM: {reason}")
                        elif "text change" in reason.lower():
                            print(f"Chunk {chunk_id}: Full scan - Text change detected at {timestamp:.2f}s: {reason}")
                        elif "click" in reason.lower():
                            print(f"Chunk {chunk_id}: Full scan - Mouse click detected at {timestamp:.2f}s: {reason}")
                        else:
                            print(f"Chunk {chunk_id}: Full scan - Change detected at {timestamp:.2f}s: {reason}")
                            
                        # Convert the frame to a PIL Image
                        img_pil = Image.fromarray(frame)
                        
                        # Add to screenshots
                        chunk_screenshots.append((img_pil, timestamp, reason))
                        print(f"Chunk {chunk_id}: Added screenshot at {timestamp:.2f}s - {reason}")
                
                # Update previous frame
                prev_frame = frame
        
        processing_time = time.time() - start_processing_time
        print(f"Chunk {chunk_id} completed in {processing_time:.2f}s with {len(chunk_screenshots)} screenshots")
        
        return chunk_id, chunk_screenshots
    
    def process_chunks_parallel(self):
        """
        Process all video chunks in parallel.
        
        Returns:
            List of screenshots (image, timestamp, reason)
        """
        # Split video into chunks
        chunks = self.split_video_into_chunks()
        
        # Create task data with chunk IDs
        task_data = [(i, start_frame, end_frame, start_time, end_time) 
                    for i, (start_frame, end_frame, start_time, end_time) in enumerate(chunks)]
        
        # Process chunks in parallel
        start_time = time.time()
        all_screenshots = []
        
        # Determine number of workers (limit based on system resources)
        max_workers = min(self.max_workers, len(chunks))
        print(f"Processing {len(chunks)} video chunks with {max_workers} workers in parallel")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(self.process_chunk, data): data[0]  # chunk_id
                for data in task_data
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    _, chunk_screenshots = future.result()
                    all_screenshots.extend(chunk_screenshots)
                    print(f"Added {len(chunk_screenshots)} screenshots from chunk {chunk_id}")
                except Exception as e:
                    print(f"Error processing chunk {chunk_id}: {e}")
        
        # Sort screenshots by timestamp
        all_screenshots.sort(key=lambda x: x[1])
        
        # Remove duplicates (screenshots that are too close in time)
        if all_screenshots:
            deduplicated = []
            last_time = -self.extractor.cooldown
            
            for img, timestamp, reason in all_screenshots:
                if timestamp - last_time >= self.extractor.cooldown:
                    deduplicated.append((img, timestamp, reason))
                    last_time = timestamp
            
            processing_time = time.time() - start_time
            print(f"Parallel processing complete: {len(deduplicated)} screenshots (after deduplication) in {processing_time:.2f}s")
            return deduplicated
        
        return []

def process_video_in_parallel(extractor, video_processor, fps, frame_count):
    """
    Process a video in parallel chunks to extract screenshots.
    
    Args:
        extractor: ScreenshotExtractor instance
        video_processor: VideoProcessor instance
        fps: Frames per second
        frame_count: Total frame count
        
    Returns:
        List of screenshots (image, timestamp, reason)
    """
    print("Starting parallel video processing...")
    print(f"Video stats: {frame_count} frames at {fps} fps = {frame_count/fps:.2f}s")
    
    # Make sure we have key timestamps - if not, create some default ones
    if not hasattr(extractor, 'key_timestamps') or not extractor.key_timestamps:
        print("Warning: No key timestamps found from Phase 1, creating default timestamps")
        # Create some evenly spaced timestamps throughout the video
        video_duration = frame_count / fps
        num_timestamps = max(5, int(video_duration / 30))  # At least 5 timestamps, or one per 30 seconds
        
        default_timestamps = []
        for i in range(num_timestamps):
            timestamp = (i * video_duration) / (num_timestamps - 1) if num_timestamps > 1 else 0
            default_timestamps.append((timestamp, "Default timestamp"))
            
        # Also add beginning and end
        default_timestamps.append((0, "Start of video"))
        default_timestamps.append((video_duration, "End of video"))
        
        # Set as key timestamps
        extractor.key_timestamps = default_timestamps
        print(f"Created {len(default_timestamps)} default timestamps")
    else:
        print(f"Using {len(extractor.key_timestamps)} timestamps from Phase 1")
        # Print first few timestamps for debugging
        for i, (timestamp, reason) in enumerate(extractor.key_timestamps[:5]):
            print(f"  Timestamp {i+1}: {timestamp:.2f}s - {reason}")
    
    # Create the processor and run parallel processing
    processor = VideoChunkProcessor(extractor, video_processor, fps, frame_count)
    screenshots = processor.process_chunks_parallel()
    
    print(f"Completed parallel video processing with {len(screenshots)} screenshots")
    return screenshots
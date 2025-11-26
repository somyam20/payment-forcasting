"""
Chunk processor module for parallel video processing
This module contains functions for processing video chunks in parallel
"""

from PIL import Image
import time
from typing import List, Tuple, Dict, Any, Optional

class ChunkProcessor:
    """
    Helper class for processing video chunks in parallel
    """
    
    @staticmethod
    def process_timestamp_chunk(chunk_data, extractor, video_processor, fps, frame_count):
        """
        Process a chunk of timestamps to extract screenshots
        
        Args:
            chunk_data: Tuple containing (chunk_id, timestamp_list)
            extractor: ScreenshotExtractor instance
            video_processor: VideoProcessor instance
            fps: Frames per second
            frame_count: Total frame count
            
        Returns:
            List of screenshots (image, timestamp, reason)
        """
        chunk_id, timestamp_list = chunk_data
        chunk_screenshots = []
        # Local set to track processed frames within this chunk
        processed_frames = set()
        
        start_time = time.time()
        print(f"Starting chunk {chunk_id} with {len(timestamp_list)} timestamps")
        
        # Initialize local state for this chunk to avoid thread conflicts
        prev_frame = None
        
        for timestamp, reason in timestamp_list:
            # Convert timestamp to frame number
            center_frame_number = int(timestamp * fps)
            
            # Special handling for scene changes - use exact frame
            if "Scene change" in reason:
                frame = video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    chunk_screenshots.append((img_pil, timestamp, reason))
                    processed_frames.add(center_frame_number)
                continue  # Skip window processing for scene changes
            
            # Special handling for AI detection - always capture exact frame
            elif "AI detected" in reason or "AI Detected" in reason:
                frame = video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    chunk_screenshots.append((img_pil, timestamp, reason))
                    processed_frames.add(center_frame_number)
            
            # Process a window around the timestamp to find the best frame
            window_size = 10  # frames to check on each side
            
            # Check frames in the window
            for offset in range(-window_size, window_size + 1):
                frame_number = center_frame_number + offset
                
                # Skip if out of range or already processed
                if (frame_number < 0 or 
                    frame_number >= frame_count or 
                    frame_number in processed_frames):
                    continue
                
                # Mark as processed in this chunk
                processed_frames.add(frame_number)
                
                # Get the frame
                frame = video_processor.get_frame_at_position(frame_number)
                if frame is None:
                    continue
                
                # Calculate exact timestamp
                exact_timestamp = frame_number / fps
                
                # Run key frame detection (but only if we have a previous frame)
                if prev_frame is not None:
                    # Use the extractor to detect key frames
                    is_key, detection_reason = extractor.is_key_frame(frame, exact_timestamp)
                    
                    if is_key:
                        # Convert frame to PIL Image
                        img_pil = Image.fromarray(frame)
                        
                        # Use detection reason if it's more specific
                        final_reason = detection_reason if detection_reason else reason
                        
                        # Add to screenshots
                        chunk_screenshots.append((img_pil, exact_timestamp, final_reason))
                        break  # Found a good frame, move to next timestamp
                
                # Update previous frame for next iteration
                prev_frame = frame
        
        processing_time = time.time() - start_time
        print(f"Chunk {chunk_id} complete: {len(chunk_screenshots)} screenshots in {processing_time:.2f}s")
        return chunk_screenshots
    
    @staticmethod
    def process_speech_chunks(chunk_data, extractor):
        """
        Process a chunk of audio segments in parallel
        
        Args:
            chunk_data: Tuple containing (chunk_id, audio_chunks)
            extractor: ScreenshotExtractor instance
            
        Returns:
            List of detected keyword timestamps and text
        """
        chunk_id, audio_chunks = chunk_data
        speech_results = []
        
        start_time = time.time()
        print(f"Processing speech chunk {chunk_id} with {len(audio_chunks)} audio segments")
        
        for chunk_time, chunk in audio_chunks:
            # Use the extractor's speech recognition capability
            result = extractor._process_single_audio_chunk(chunk_time, chunk)
            if result:
                speech_results.extend(result)
        
        processing_time = time.time() - start_time
        print(f"Speech chunk {chunk_id} complete: {len(speech_results)} results in {processing_time:.2f}s")
        return speech_results
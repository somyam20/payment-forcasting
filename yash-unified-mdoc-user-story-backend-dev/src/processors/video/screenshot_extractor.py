import cv2
import numpy as np
import pytesseract
import speech_recognition as sr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import tempfile
import os
import re
from pydub import AudioSegment
import io
import concurrent.futures
import time
import math
from typing import List, Tuple, Dict, Any, Optional
from ..audio.whisper_processor import WhisperProcessor, get_optimized_whisper_processor
from ..utils.face_pii import FastFaceBlurProcessor
# Import our parallel processing modules
from ..parallel.parallel_processor import ParallelProcessor
from ..utils.chunk_processor import ChunkProcessor
from ..parallel.parallel_video_processor import ParallelVideoProcessor
from ..parallel.parallel_extractor import ParallelExtractor
from ..parallel.parallel_speech import process_speech_in_parallel
from ..parallel.parallel_video_chunks import process_video_in_parallel
import logging

from ...utils.logger_config import setup_logger

setup_logger()

# Conditionally import OpenAI analyzer and client
try:
    from ...utils.openai_analyzer import analyze_speech_for_screenshot_moments, OPENAI_AVAILABLE, client
except ImportError:
    OPENAI_AVAILABLE = False
    client = None
    def analyze_speech_for_screenshot_moments(speech_segments, teams_llm_config):
        return []  # Fallback function

class ScreenshotExtractor:
    def __init__(self, threshold=25, min_area=2000, text_change_threshold=10, cooldown=3, 
                 ssim_threshold=0.85, keyword_trigger=True, mouse_tracking=True, 
                 use_ai_speech_analysis=False, parallel_processing=True, use_ssim=True, 
                 detection_mode='advanced'):
        """
        Initialize the screenshot extractor with detection parameters.
        
        Args:
            threshold (int): Threshold for pixel difference detection (0-255).
            min_area (int): Minimum area size for considering a change significant.
            text_change_threshold (int): Threshold for detecting text changes.
            cooldown (float): Minimum time in seconds between screenshots.
            ssim_threshold (float): Threshold for structural similarity (lower means more sensitivity).
            keyword_trigger (bool): Whether to use keyword-based triggering.
            mouse_tracking (bool): Whether to track mouse cursor and interactions.
            use_ai_speech_analysis (bool): Whether to use OpenAI for intelligent speech analysis.
            parallel_processing (bool): Whether to use parallel processing for improved performance.
            use_ssim (bool): Legacy parameter for backwards compatibility.
            detection_mode (str): Detection mode - 'basic' (excludes scene changes) or 'advanced' (includes all methods).
        """
        self.threshold = threshold
        self.min_area = min_area
        self.text_change_threshold = text_change_threshold
        self.cooldown = cooldown
        self.ssim_threshold = ssim_threshold
        self.keyword_trigger = keyword_trigger
        self.mouse_tracking = mouse_tracking
        self.parallel_processing = parallel_processing
        self.detection_mode = detection_mode
        
        # Initialize detection result lists
        self.keyword_timestamps = []
        self.scene_change_timestamps = []
        self.mouse_timestamps = []  # Added to store mouse interaction timestamps
        
        # Initialize parallel processor if enabled
        if self.parallel_processing:
            self.parallel_processor = ParallelProcessor()
        
        # Check if OpenAI is available for AI speech analysis
        if use_ai_speech_analysis and not OPENAI_AVAILABLE:
            logging.info("Warning: OpenAI API is not available. Disabling AI speech analysis.")
            self.use_ai_speech_analysis = False
        else:
            self.use_ai_speech_analysis = use_ai_speech_analysis
        
        # For tracking state between frames
        self.prev_frame = None
        self.prev_text = None
        self.prev_gray = None
        self.prev_cursor_pos = None
        self.last_screenshot_time = -cooldown  # Allow immediate screenshot on start
        
        # Audio processing and speech recognition - Using Whisper for better accuracy
        self.recognizer = sr.Recognizer()
        self.temp_audio_file = None
        self.audio_chunks = []
        self.speech_timestamps = []
        
        # Initialize Whisper processor for superior speech recognition
        try:
            self.whisper_processor = get_optimized_whisper_processor()
            logging.info("Whisper processor initialized for high-quality speech recognition")
            self.use_whisper = True
        except Exception as e:
            logging.exception(f"Whisper initialization failed, falling back to Google Speech Recognition: {e}")
            self.whisper_processor = None
            self.use_whisper = False
        
        # Keywords to detect for triggering screenshots
        self.trigger_keywords = [
            r'\bclick\b', r'\bselect\b', r'\bnavigate\b', r'\bchoose\b', r'\btap\b',
            r'\bopen\b', r'\benter\b', r'\btype\b', r'\bfill\b', r'\bscroll\b',
            r'\bgo to\b', r'\bmove to\b', r'\bdrag\b', r'\bdrop\b', r'\bcheck\b',
            r'\bview\b', r'\bsearch\b', r'\bfind\b', r'\blook at\b', r'\bnotice\b'
        ]
        
        # Queue to store recent frames for intelligent selection
        self.frame_buffer = []
        self.buffer_size = 15  # Store frames for intelligent selection
        
        # For two-phase processing
        self.key_timestamps = []  # Store timestamps of interest
        self.scene_change_timestamps = []  # Store scene change timestamps
        self.keyword_timestamps = []  # Store keyword trigger timestamps
        self.ai_timestamps = []  # Store AI-detected timestamps
        self.processed_timestamps = set()  # Track which timestamps we've processed
    
    def extract_audio_from_video(self, video_path):
        """
        Extract audio from video file for speech recognition.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            bool: True if audio extraction was successful, False otherwise.
        """
        try:
            # First try: Use ffmpeg if available
            import subprocess
            ffmpeg_available = False
            
            try:
                # Check if ffmpeg is available
                subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                ffmpeg_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                logging.exception("FFmpeg not available. Will try alternative methods.")
                ffmpeg_available = False
            
            if ffmpeg_available:
                # Create a temporary file to store audio
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                self.temp_audio_file = temp_audio.name
                temp_audio.close()
                
                # Extract audio using ffmpeg with optimized settings
                # -y: Overwrite output file without asking
                # -vn: Disable video recording
                # -ar 16000: Set audio sampling rate to 16KHz (good for speech)
                # -ac 1: Set audio channels to 1 (mono)
                # -q:a 0: Use highest quality audio
                result = subprocess.run(['ffmpeg', '-i', video_path, '-vn', '-ar', '16000', 
                                        '-ac', '1', '-q:a', '0', self.temp_audio_file, '-y'],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Check if the audio file was created successfully
                if os.path.exists(self.temp_audio_file) and os.path.getsize(self.temp_audio_file) > 0:
                    # Load the audio file for processing
                    audio = AudioSegment.from_wav(self.temp_audio_file)
                    
                    # For long videos, use longer chunks to reduce processing time
                    # This will decrease accuracy slightly but improve performance significantly
                    if len(audio) > 300000:  # 5 minutes
                        chunk_length_ms = 30000  # 30 seconds for long videos
                    else:
                        chunk_length_ms = 15000  # 15 seconds for short videos
                    
                    # Split into chunks for speech recognition
                    for i in range(0, len(audio), chunk_length_ms):
                        chunk = audio[i:i+chunk_length_ms]
                        self.audio_chunks.append((i / 1000, chunk))  # Store start time and chunk
                    
                    logging.info(f"Audio extracted successfully with FFmpeg - created {len(self.audio_chunks)} chunks")
                    return True
                else:
                    logging.info("Audio extraction with FFmpeg failed. Trying alternative methods.")
            
            # Second try: Use local Whisper processor for high-quality transcription
            local_whisper_succeeded = False
            if self.use_whisper and self.whisper_processor:
                logging.info("Using local Whisper processor for high-quality speech recognition")
                
                try:
                    # Use Whisper processor to extract speech segments with precise timestamps
                    speech_segments = self.whisper_processor.extract_speech_segments(video_path)
                    
                    if speech_segments:
                        # Store speech segments with precise timestamps
                        self.speech_timestamps = speech_segments
                        self.speech_segments = speech_segments  # Store for sliding window analysis
                        
                        # Create audio chunks for compatibility with existing system
                        self.audio_chunks = []
                        for timestamp, text in speech_segments:
                            # Create a minimal audio segment for interface compatibility
                            empty_audio = AudioSegment.silent(duration=1000)  # 1 second silence
                            self.audio_chunks.append((timestamp, empty_audio))
                        
                        logging.info(f"Whisper processing successful: {len(speech_segments)} speech segments with precise timestamps")
                        local_whisper_succeeded = True
                        return True
                    else:
                        logging.info("No speech detected in video by Whisper")
                        
                except Exception as e:
                    logging.exception(f"Local Whisper processing failed: {e}")
            
            # Third try: Fallback to OpenAI Whisper API only if local Whisper failed or is not available
            # Skip API fallback if local Whisper successfully processed the video
            if not local_whisper_succeeded and OPENAI_AVAILABLE and client is not None:
                # Calculate video duration using OpenCV
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Calculate duration
                if fps > 0 and frame_count > 0:
                    video_duration = frame_count / fps
                else:
                    video_duration = 180  # Default to 3 minutes if we can't determine
                
                logging.info(f"Fallback: Using OpenAI Whisper API for transcription (duration: {video_duration:.2f}s)")
                
                try:
                    # Use our dedicated Whisper helper function
                    logging.info("Uploading video to Whisper API for transcription...")
                    
                    # Import our Whisper transcription helper
                    from ...utils.media_utils import transcribe_with_whisper
                    
                    # Use standard OpenAI for Whisper transcription
                    transcription_text = transcribe_with_whisper(video_path)
                    logging.info(f"Whisper transcription successful: {len(transcription_text)} characters")
                    
                    # Create artificial "chunks" every 15 seconds
                    self.speech_timestamps = []
                    num_chunks = min(12, max(1, int(video_duration / 15)))
                    chunk_length_ms = 15000  # 15 seconds per chunk
                    
                    # Create empty audio chunks (we don't have actual audio data)
                    for i in range(num_chunks):
                        timestamp = i * (chunk_length_ms / 1000)
                        
                        # Create a timestamp with the portion of text
                        # For simplicity, divide the text equally among chunks
                        start_idx = int(i * len(transcription_text) / num_chunks)
                        end_idx = int((i + 1) * len(transcription_text) / num_chunks)
                        chunk_text = transcription_text[start_idx:end_idx]
                        
                        # Store timestamp and text
                        if chunk_text.strip():
                            self.speech_timestamps.append((timestamp, chunk_text))
                            
                            # Create a dummy audio chunk (we'll need this for the interface)
                            empty_audio = AudioSegment.silent(duration=1000)  # 1 sec silence
                            self.audio_chunks.append((timestamp, empty_audio))
                    
                    logging.info(f"Created {len(self.audio_chunks)} simulated chunks from Whisper transcription")
                    return True
                except Exception as e:
                    logging.exception(f"Error using Whisper API: {e}")
            
            # If we reach here, both methods failed
            logging.info("Could not extract audio: FFmpeg not available and OpenAI Whisper API failed")
            return False
                
        except Exception as e:
            logging.exception(f"Error extracting audio: {e}")
            return False
    
    def detect_keywords_in_speech(self, start_time, end_time):
        """
        Process speech for keyword detection within a time range.
        
        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            
        Returns:
            tuple: (contains_keyword, detected_phrase) indicating if a keyword was found
                  and what phrase was detected.
        """
        # Skip if speech recognition isn't enabled or we have no chunks
        if not self.keyword_trigger or not self.audio_chunks:
            return False, ""
            
        # Check if we've already detected keywords at timestamps close to this range
        # This prevents redundant processing of the same speech segments
        for timestamp, text in self.speech_timestamps:
            # If we already detected speech within 1 second of this range
            if abs(timestamp - start_time) < 1.0 or abs(timestamp - end_time) < 1.0:
                # Check if this text contains keywords
                for pattern in self.trigger_keywords:
                    if re.search(pattern, text.lower()):
                        return True, f"Previously detected: {text}"
        
        # Find chunks that overlap with the given time range
        relevant_chunks = []
        for chunk_time, chunk in self.audio_chunks:
            chunk_end_time = chunk_time + len(chunk) / 1000.0
            if chunk_time <= end_time and chunk_end_time >= start_time:
                relevant_chunks.append((chunk_time, chunk))
        
        # Skip if no relevant chunks found
        if not relevant_chunks:
            return False, ""
        
        # Process each relevant chunk - with a maximum limit to prevent excessive processing
        max_chunks_to_process = 1  # Only process the most relevant chunk to save time
        for chunk_time, chunk in relevant_chunks[:max_chunks_to_process]:
            # Convert chunk to format needed by SpeechRecognition
            chunk_file = io.BytesIO()
            chunk.export(chunk_file, format="wav")
            chunk_file.seek(0)
            
            try:
                # Use speech recognition to convert audio to text
                with sr.AudioFile(chunk_file) as source:
                    audio_data = self.recognizer.record(source)
                    
                    # Try with speech recognition services
                    text = ""
                    
                    # First try Google's API if available
                    try:
                        text = self.recognizer.recognize_google(audio_data)
                    except (sr.RequestError, AttributeError, Exception) as e:
                        logging.exception(f"Google speech recognition failed: {e}")
                        
                        # If Google fails and OpenAI is available, try using Whisper
                        if OPENAI_AVAILABLE and client is not None:
                            try:
                                # Export the audio data to a temporary file
                                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                temp_file.close()
                                
                                # Use Python's wave to write the audio data
                                import wave
                                with wave.open(temp_file.name, 'wb') as wf:
                                    wf.setnchannels(1)  # Mono
                                    wf.setsampwidth(2)  # 16-bit
                                    wf.setframerate(16000)  # 16kHz
                                    wf.writeframes(audio_data.get_raw_data())
                                
                                # Use our Whisper API utility function
                                from ...utils.media_utils import transcribe_with_whisper
                                
                                # This will always use standard OpenAI for Whisper transcription
                                text = transcribe_with_whisper(temp_file.name)
                                logging.info(f"Used Whisper API for speech recognition")
                                
                                # Clean up temp file
                                try:
                                    os.unlink(temp_file.name)
                                except:
                                    pass
                                    
                            except Exception as whisper_err:
                                logging.exception(f"Whisper API speech recognition failed: {whisper_err}")
                                return False, ""
                    
                    if not text:
                        logging.info("Speech recognition failed with all methods")
                        return False, ""
                    
                    # Store the speech text for reference regardless of keywords
                    timestamp = chunk_time + (end_time - start_time) / 2
                    self.speech_timestamps.append((timestamp, text))
                    
                    # Check for trigger keywords
                    for pattern in self.trigger_keywords:
                        if re.search(pattern, text.lower()):
                            return True, text
            except Exception as e:
                logging.error("Speech recognition error: %s", e)
        
        return False, ""
    
    def detect_cursor_movement(self, current_frame, prev_frame):
        """
        Detect cursor movement and potential click actions with enhanced robustness.
        
        Args:
            current_frame (numpy.ndarray): Current video frame.
            prev_frame (numpy.ndarray): Previous video frame.
            
        Returns:
            tuple: (click_detected, cursor_position) where click_detected is a boolean
                  and cursor_position is a tuple (x, y) or None if no cursor is detected.
        """
        if not self.mouse_tracking:
            return False, None
            
        # Convert frames to grayscale for processing
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        # Store previous cursor shapes for tracking cursor state changes
        if not hasattr(self, 'cursor_history'):
            self.cursor_history = []
            
        # Initialize cursor tracking parameters if not already done
        if not hasattr(self, 'cursor_params'):
            self.cursor_params = {
                'min_area': 5,              # Minimum area of cursor blob
                'max_area': 500,            # Maximum area of cursor blob 
                'brightness_threshold': 220, # Threshold for bright cursor detection
                'movement_threshold': 10,    # Max pixels movement to consider static
                'min_cursor_aspect_ratio': 0.2, # Min aspect ratio for cursor (not too elongated)
                'max_cursor_aspect_ratio': 5.0, # Max aspect ratio for cursor
                'click_region_size': 60,    # Size of region to check for changes around cursor
                'click_change_threshold': 120, # Min pixel changes to detect a click
                'cursor_shapes': []          # Store different cursor shapes/templates
            }
        
        # Multi-method approach to cursor detection
        cursor_candidates = []
        
        # Method 1: Bright spot detection (for white/light cursors)
        _, bright_thresh = cv2.threshold(current_gray, self.cursor_params['brightness_threshold'], 255, cv2.THRESH_BINARY)
        
        # Apply a slight blur to reduce noise before contour detection
        bright_thresh = cv2.GaussianBlur(bright_thresh, (3, 3), 0)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape to find cursor-like objects
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.cursor_params['min_area'] < area < self.cursor_params['max_area']:
                # Check aspect ratio to filter out unlikely cursor shapes
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Skip if width or height is zero to avoid division by zero
                if w == 0 or h == 0:
                    continue
                    
                aspect_ratio = float(w) / h
                
                # Cursors typically have an aspect ratio close to 1 (not too elongated)
                if (self.cursor_params['min_cursor_aspect_ratio'] < aspect_ratio < 
                    self.cursor_params['max_cursor_aspect_ratio']):
                    
                    # Calculate centroid
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Save additional properties for better cursor tracking
                        cursor_candidates.append({
                            'position': (cx, cy),
                            'area': area,
                            'contour': cnt,
                            'bounding_rect': (x, y, w, h),
                            'confidence': 0.0  # Will be updated based on tracking
                        })
        
        # Method 2: Frame differencing to detect moving objects (for dark cursors)
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate absolute difference between frames
            frame_diff = cv2.absdiff(current_gray, prev_gray)
            
            # Threshold the difference image
            _, motion_thresh = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            motion_thresh = cv2.morphologyEx(motion_thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours in motion mask
            motion_contours, _ = cv2.findContours(motion_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter motion contours by size
            for cnt in motion_contours:
                area = cv2.contourArea(cnt)
                if self.cursor_params['min_area'] < area < self.cursor_params['max_area']:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Skip if width or height is zero
                    if w == 0 or h == 0:
                        continue
                        
                    aspect_ratio = float(w) / h
                    
                    # Cursor shapes are typically not too elongated
                    if (self.cursor_params['min_cursor_aspect_ratio'] < aspect_ratio < 
                        self.cursor_params['max_cursor_aspect_ratio']):
                        
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Add to candidates if not already added from bright detection
                            duplicate = False
                            for cand in cursor_candidates:
                                # Check if we already found this candidate
                                dist = np.sqrt((cx - cand['position'][0])**2 + 
                                              (cy - cand['position'][1])**2)
                                if dist < 10:  # Close enough to be the same object
                                    duplicate = True
                                    # Increase confidence if detected by both methods
                                    cand['confidence'] += 0.3
                                    break
                                    
                            if not duplicate:
                                cursor_candidates.append({
                                    'position': (cx, cy),
                                    'area': area, 
                                    'contour': cnt,
                                    'bounding_rect': (x, y, w, h),
                                    'confidence': 0.2  # Lower initial confidence for motion-only detection
                                })
        
        # Use tracking history to improve detection
        if self.prev_cursor_pos is not None:
            for cand in cursor_candidates:
                # Check distance from previous cursor position
                dist = np.sqrt((cand['position'][0] - self.prev_cursor_pos[0])**2 + 
                              (cand['position'][1] - self.prev_cursor_pos[1])**2)
                
                # Add confidence based on distance (closer to previous position = higher confidence)
                # Use a sigmoid-like function to map distance to confidence
                distance_confidence = 0.5 / (1 + 0.01 * dist**2)
                cand['confidence'] += distance_confidence
        
        # Select best cursor candidate
        cursor_pos = None
        click_detected = False
        best_confidence = 0.0
        
        if cursor_candidates:
            # Sort by confidence
            cursor_candidates.sort(key=lambda c: c['confidence'], reverse=True)
            best_candidate = cursor_candidates[0]
            
            # Only accept if confidence is high enough
            if best_candidate['confidence'] > 0.1:
                cursor_pos = best_candidate['position']
                
                # Check for click events
                if self.prev_cursor_pos is not None:
                    # Calculate cursor movement
                    cursor_movement = np.sqrt((cursor_pos[0] - self.prev_cursor_pos[0])**2 + 
                                             (cursor_pos[1] - self.prev_cursor_pos[1])**2)
                    
                    # Store cursor shape information
                    x, y, w, h = best_candidate['bounding_rect']
                    if current_gray.shape[0] > y + h and current_gray.shape[1] > x + w:
                        cursor_roi = current_gray[y:y+h, x:x+w]
                        
                        # Add to cursor history (limited to last 3 shapes)
                        if len(self.cursor_history) >= 3:
                            self.cursor_history.pop(0)
                        self.cursor_history.append({
                            'roi': cursor_roi.copy() if cursor_roi.size > 0 else None,
                            'position': cursor_pos,
                            'area': best_candidate['area']
                        })
                    
                    # Click detection method 1: Cursor hasn't moved much (static cursor)
                    if cursor_movement < self.cursor_params['movement_threshold']:
                        # Check for UI changes around the cursor (potential click effects)
                        region_size = self.cursor_params['click_region_size']
                        x1 = max(0, cursor_pos[0] - region_size)
                        y1 = max(0, cursor_pos[1] - region_size)
                        x2 = min(current_gray.shape[1], cursor_pos[0] + region_size)
                        y2 = min(current_gray.shape[0], cursor_pos[1] + region_size)
                        
                        if self.prev_gray is not None and current_gray.shape == self.prev_gray.shape:
                            # Check region around cursor for changes
                            region_current = current_gray[y1:y2, x1:x2]
                            region_prev = self.prev_gray[y1:y2, x1:x2]
                            
                            if region_current.size > 0 and region_prev.size > 0 and region_current.shape == region_prev.shape:
                                # Enhanced difference detection with adaptive thresholding
                                diff = cv2.absdiff(region_current, region_prev)
                                
                                # Use adaptive thresholding for better change detection
                                diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
                                _, diff_thresh = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_BINARY)
                                
                                # Apply morphological operations to clean up noise
                                kernel = np.ones((3, 3), np.uint8)
                                diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
                                
                                # Count significant changes
                                changes = np.sum(diff_thresh > 0)
                                
                                if changes > self.cursor_params['click_change_threshold']:
                                    click_detected = True
                    
                    # Click detection method 2: Check for cursor shape changes (cursor icon changing)
                    if len(self.cursor_history) >= 2 and not click_detected:
                        current_shape = self.cursor_history[-1]
                        prev_shape = self.cursor_history[-2]
                        
                        # Check for significant area change (cursor shape changing)
                        if current_shape['roi'] is not None and prev_shape['roi'] is not None:
                            # Resize to same dimensions for comparison
                            if current_shape['roi'].shape != prev_shape['roi'].shape and \
                               current_shape['roi'].size > 0 and prev_shape['roi'].size > 0:
                                try:
                                    # Resize larger to smaller to avoid information loss
                                    if current_shape['roi'].size > prev_shape['roi'].size:
                                        resized_current = cv2.resize(current_shape['roi'], 
                                                                    (prev_shape['roi'].shape[1], prev_shape['roi'].shape[0]))
                                        shape_diff = cv2.absdiff(resized_current, prev_shape['roi'])
                                    else:
                                        resized_prev = cv2.resize(prev_shape['roi'], 
                                                                 (current_shape['roi'].shape[1], current_shape['roi'].shape[0]))
                                        shape_diff = cv2.absdiff(current_shape['roi'], resized_prev)
                                    
                                    # Calculate shape difference
                                    shape_change = np.mean(shape_diff)
                                    
                                    # Significant shape change could indicate cursor icon changing (e.g., pointer to hand)
                                    if shape_change > 30:
                                        click_detected = True
                                except Exception as e:
                                    # Skip shape comparison if resize fails
                                    pass
                            
                            # Also check for area changes
                            area_change_pct = abs(current_shape['area'] - prev_shape['area']) / max(1, prev_shape['area'])
                            if area_change_pct > 0.3:  # Area changed by more than 30%
                                click_detected = True
        
        # Update state for next frame
        self.prev_cursor_pos = cursor_pos
        self.prev_gray = current_gray.copy()
        
        return click_detected, cursor_pos
    
    def calculate_ssim(self, frame1, frame2):
        """
        Calculate the structural similarity between two frames.
        
        Args:
            frame1 (numpy.ndarray): First frame.
            frame2 (numpy.ndarray): Second frame.
            
        Returns:
            float: SSIM index between the two frames (0-1, higher is more similar).
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score
    
    def is_key_frame(self, frame, timestamp):
        """
        Determine if a frame is a key frame that should be captured as a screenshot.
        
        Args:
            frame (numpy.ndarray): The current video frame.
            timestamp (float): The timestamp of the frame in seconds.
            
        Returns:
            tuple: (is_key_frame, reason) where is_key_frame is a boolean and reason
                  is a string describing why the frame was considered key.
        """
        # Store the current timestamp for use in detection methods
        self.current_timestamp = timestamp
        
        # Only use cooldown AFTER a trigger is detected
        # This prevents capturing frames at constant intervals
        # But still ensures we don't capture too many screenshots during rapid changes
        if timestamp - self.last_screenshot_time < self.cooldown:
            # Still add to frame buffer even during cooldown
            self.frame_buffer.append((frame.copy(), timestamp))
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            # But we still check for events, so we can capture important screenshots
            # even during cooldown if an event is significant enough
        
        # Initialize result - no longer defaulting to key frames
        is_key_frame = False
        reason = "No significant change"
        
        # Add current frame to buffer
        self.frame_buffer.append((frame.copy(), timestamp))
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # 1. Check for mouse interaction/clicks
        click_detected, cursor_pos = False, None
        if self.mouse_tracking and self.prev_frame is not None:
            click_detected, cursor_pos = self.detect_cursor_movement(frame, self.prev_frame)
            if click_detected:
                is_key_frame = True
                reason = f"Mouse click detected at position {cursor_pos}"
                # Also store this in our mouse_timestamps for the two-phase process
                self.mouse_timestamps.append((timestamp, reason))
        
        # 2. Check for speech/keyword triggers
        if self.keyword_trigger and not is_key_frame:
            # We check speech for a window around the current timestamp
            # This accounts for potential delay between speech and visual changes
            speech_window_start = max(0, timestamp - 3)
            speech_window_end = timestamp + 1
            
            keyword_detected, detected_text = self.detect_keywords_in_speech(speech_window_start, speech_window_end)
            if keyword_detected:
                is_key_frame = True
                reason = f"Keyword trigger: {detected_text}"
        
        # 3. Check for structural similarity (scene change detection)
        # Only do this if we have a previous frame, no trigger yet, and advanced mode is enabled
        detection_mode = getattr(self, 'detection_mode', 'advanced')  # Default to advanced for backwards compatibility
        
        # Skip scene change detection in basic mode
        if detection_mode == 'basic':
            # In basic mode, skip all scene change detection
            pass
        elif self.prev_frame is not None and not is_key_frame:
            # Resize frame if needed for performance
            if frame.shape[0] > 720:
                scale = 720 / frame.shape[0]
                width = int(frame.shape[1] * scale)
                frame_resized = cv2.resize(frame, (width, 720))
                prev_resized = cv2.resize(self.prev_frame, (width, 720))
            else:
                frame_resized = frame
                prev_resized = self.prev_frame
                
            # Calculate structural similarity
            similarity = self.calculate_ssim(frame_resized, prev_resized)
            
            # Low similarity means significant change - a major scene change
            # Make this threshold more strict to only capture major UI changes
            if similarity < self.ssim_threshold:
                is_key_frame = True
                reason = f"Major scene change detected (SSIM={similarity:.3f})"
            else:
                # 4. Check for significant visual changes - only for major content changes
                # We increase the threshold to only detect significant changes
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
                prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_RGB2GRAY)
                
                # Compute absolute difference
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
                
                # Dilate threshold image to fill in holes
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check for significant contours - we require more significant changes
                # Look for larger contours or multiple significant changes
                significant_contours = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > self.min_area:
                        significant_contours += 1
                        # If we have one very large change or multiple significant changes
                        if area > self.min_area * 2 or significant_contours >= 3:
                            is_key_frame = True
                            reason = "Major visual change detected"
                            break
                        
                # 5. Check for text changes if no visual change detected
                # Only detect MAJOR text changes - like new paragraphs or headings, not minor edits
                if not is_key_frame:
                    try:
                        # Use a region of interest for OCR to improve performance
                        height, width = gray.shape
                        roi = gray[int(height*0.1):int(height*0.9), int(width*0.1):int(width*0.9)]
                        
                        # Use Pytesseract to extract text
                        current_text = pytesseract.image_to_string(roi)
                        
                        # Only detect significant text changes
                        if self.prev_text is not None and current_text != self.prev_text:
                            # More strict metric: percentage of changed characters plus length change
                            max_len = max(len(current_text), len(self.prev_text))
                            if max_len > 0:  # Avoid division by zero
                                # Count changed characters
                                diff_count = sum(1 for a, b in zip(current_text, self.prev_text) if a != b)
                                # Add difference in length
                                diff_count += abs(len(current_text) - len(self.prev_text))
                                # Calculate percentage change
                                percent_change = (diff_count / max_len) * 100
                                
                                # Only trigger on significant text changes (>30% change)
                                if percent_change > 30:
                                    is_key_frame = True
                                    reason = "Significant text content change detected"
                        
                        self.prev_text = current_text
                    except Exception as e:
                        # OCR can fail for various reasons, just log and continue
                        logging.error("OCR error: %s", e)
        elif self.prev_frame is None:
            # First frame is always a key frame
            is_key_frame = True
            reason = "First frame"
            
        # Update state
        self.prev_frame = frame.copy()
        
        if is_key_frame:
            # Only update the cooldown timer if we've decided this is a key frame
            self.last_screenshot_time = timestamp
            
            # Intelligent frame selection: instead of current frame,
            # select best frame from buffer (usually 1-2 frames after a trigger)
            if len(self.frame_buffer) > 2:
                # For clicks/keywords, take frame a bit after the event
                # to capture the result of the action
                delay_frames = min(2, len(self.frame_buffer) - 1)
                selected_frame, selected_timestamp = self.frame_buffer[-delay_frames]
                return True, f"{reason} (intelligent selection)"
            
        return is_key_frame, reason
        
    def extract_all_speech_keywords(self, teams_llm_config):
        """
        Pre-process all audio chunks to find speech keywords in one pass.
        Uses parallel processing for significantly faster results when enabled.
        
        Returns:
            list: List of tuples (timestamp, text) where keywords were found
        """
        start_time = time.time()
        # Preserve existing speech_timestamps if they were already populated (e.g., by Whisper in extract_audio_from_video)
        if not hasattr(self, 'speech_timestamps') or not self.speech_timestamps:
            self.speech_timestamps = []
        # Always reset keyword and AI timestamps as they need to be recalculated
        self.keyword_timestamps = []
        self.ai_timestamps = []
        
        # Skip if no audio chunks or all speech recognition methods disabled
        # But if we already have speech_timestamps, we should still process them for keywords/AI
        if not self.audio_chunks or (not self.keyword_trigger and not self.use_ai_speech_analysis):
            # If we have speech_timestamps but no audio_chunks, we can still return them
            if self.speech_timestamps:
                logging.info("Using pre-populated speech_timestamps (%d segments)", len(self.speech_timestamps))
                return self.speech_timestamps
            return []
            
        logging.info("Pre-processing %d audio chunks for speech...", len(self.audio_chunks))
        
        # Determine if we should use parallel processing
        if self.parallel_processing:
            # Use parallel speech processing for better performance
            logging.info("Using PARALLEL processing for %d speech chunks", len(self.audio_chunks))
            from ..parallel.parallel_speech import process_speech_in_parallel
            
            # Set up trigger keywords if enabled
            trigger_keywords = self.trigger_keywords if self.keyword_trigger else None
            
            # Process all audio chunks in parallel
            speech_results, keyword_results = process_speech_in_parallel(
                self.audio_chunks,
                self.recognizer,
                trigger_keywords,
                self.use_ai_speech_analysis and OPENAI_AVAILABLE and client is not None
            )
            
            # Store the results
            self.speech_timestamps = speech_results
            
            # Process keyword results - add buffer timestamps if needed
            if keyword_results:
                for timestamp, text in keyword_results:
                    # Add buffer timestamps around the keyword for context
                    buffer_before = 2.0  # seconds before keyword
                    buffer_after = 5.0   # seconds after keyword
                    
                    for t in [timestamp - buffer_before, timestamp, timestamp + buffer_after]:
                        if t >= 0:  # Ensure valid timestamp
                            self.keyword_timestamps.append((t, text))
                            break  # First detection is enough
            
            logging.info("Parallel speech processing complete with %d segments and %d keywords", len(speech_results), len(keyword_results))
        else:
            # Use sequential processing (original method)
            logging.info("Using SEQUENTIAL processing for %d speech chunks", len(self.audio_chunks))
            
            # First pass: extract all speech to text sequentially
            for chunk_index, (chunk_time, chunk) in enumerate(self.audio_chunks):
                # Convert chunk to format needed by SpeechRecognition
                chunk_file = io.BytesIO()
                chunk.export(chunk_file, format="wav")
                chunk_file.seek(0)
                
                try:
                    # Use speech recognition to convert audio to text
                    with sr.AudioFile(chunk_file) as source:
                        audio_data = self.recognizer.record(source)
                        
                        # Try with speech recognition services
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
                            
                            # Use Azure AI Speech directly with language setting
                            azure_speech_client = get_azure_speech_client()
                            if azure_speech_client.is_available():
                                # Get language from session state if available
                                import streamlit as st
                                language = getattr(st.session_state, 'speech_language', 'en-IN')
                                result = azure_speech_client.transcribe_audio_segment(audio_segment, language)
                                text = result.get('text', '').strip()
                                logging.debug("Chunk %d/%d: Recognized speech with Azure AI Speech (%s)", chunk_index+1, len(self.audio_chunks), language)
                            else:
                                raise Exception("Azure AI Speech not available")
                                
                        except Exception as azure_speech_err:
                            logging.warning("Azure AI Speech failed for chunk %d: %s", chunk_index+1, azure_speech_err)
                            
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
                                    logging.debug("Chunk %d/%d: Recognized speech with %s Whisper", chunk_index+1, len(self.audio_chunks), azure_client.service_type)
                                else:
                                    raise Exception("Azure OpenAI Whisper not available")
                                    
                            except Exception as whisper_service_err:
                                logging.debug("Azure OpenAI Whisper failed for chunk %d: %s", chunk_index+1, whisper_service_err)
                            
                            # Fallback to Google Speech Recognition
                            try:
                                text = self.recognizer.recognize_google(audio_data)
                                logging.debug("Chunk %d/%d: Recognized speech with Google (fallback)", chunk_index+1, len(self.audio_chunks))
                            except (sr.RequestError, AttributeError, Exception) as e:
                                logging.debug("Google speech recognition failed for chunk %d: %s", chunk_index+1, e)
                                
                                # Final fallback to OpenAI Whisper API if available
                                if OPENAI_AVAILABLE and client is not None:
                                    try:
                                        # Export the audio data to a temporary file
                                        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                                        temp_file.close()
                                        
                                        # Use Python's wave to write the audio data
                                        import wave
                                        with wave.open(temp_file.name, 'wb') as wf:
                                            wf.setnchannels(1)  # Mono
                                            wf.setsampwidth(2)  # 16-bit
                                            wf.setframerate(16000)  # 16kHz
                                            wf.writeframes(audio_data.get_raw_data())
                                        
                                        # Use our Whisper API utility function
                                        from ...utils.media_utils import transcribe_with_whisper
                                        
                                        # This will always use standard OpenAI for Whisper transcription
                                        text = transcribe_with_whisper(temp_file.name)
                                        logging.debug("Chunk %d/%d: Recognized speech with Whisper API (final fallback)", chunk_index+1, len(self.audio_chunks))
                                        
                                        # Clean up temp file
                                        try:
                                            os.unlink(temp_file.name)
                                        except:
                                            pass
                                            
                                    except Exception as whisper_err:
                                        logging.warning("All speech recognition methods failed for chunk %d: %s", chunk_index+1, whisper_err)
                        
                        if text:
                            # Store all transcribed speech
                            timestamp = chunk_time + len(chunk) / 2000.0  # Middle of the chunk
                            self.speech_timestamps.append((timestamp, text))
                            
                            # Check for trigger keywords if enabled
                            if self.keyword_trigger:
                                for pattern in self.trigger_keywords:
                                    if re.search(pattern, text.lower()):
                                        logging.info("Keyword found at %.2fs: %s", timestamp, text)
                                        # Store timestamps with buffer before and after
                                        # This ensures we capture context around the keyword
                                        buffer_before = 2.0  # seconds before keyword
                                        buffer_after = 5.0   # seconds after keyword
                                        
                                        for t in [timestamp - buffer_before, timestamp, timestamp + buffer_after]:
                                            if t >= 0:  # Ensure valid timestamp
                                                self.keyword_timestamps.append((t, f"Keyword trigger: {text}"))
                                                break  # First one detection is enough
                        
                except Exception as e:
                    logging.error("Speech recognition error: %s", e)
        
        speech_time = time.time() - start_time
        logging.info("Speech processing completed in %.2fs", speech_time)
        
        # Use AI to analyze the complete transcript if enabled
        if self.use_ai_speech_analysis and self.speech_timestamps:
            ai_start_time = time.time()
            logging.info("Using OpenAI to analyze %d speech segments...", len(self.speech_timestamps))
            try:
                # Use OpenAI to analyze all speech segments in one pass
                ai_results = analyze_speech_for_screenshot_moments(self.speech_timestamps, teams_llm_config)
                
                # Make sure all AI results are properly labeled to be filterable in the UI
                self.ai_timestamps = []
                for timestamp, reason in ai_results:
                    # Ensure all AI reasons start with "AI Detected" for filtering
                    if not reason.startswith("AI detected"):
                        reason = f"AI detected: {reason}"
                    self.ai_timestamps.append((timestamp, reason))
                    
                logging.info("AI identified %d potential screenshot moments", len(self.ai_timestamps))
            except Exception as e:
                logging.error("Error during AI speech analysis: %s", e)
                
            ai_time = time.time() - ai_start_time
            logging.info("AI analysis completed in %.2fs", ai_time)
                
        # Print results
        if self.keyword_trigger:
            logging.info("Found %d keyword-triggered timestamps", len(self.keyword_timestamps))
        if self.use_ai_speech_analysis:
            logging.info("Found %d AI-triggered timestamps", len(self.ai_timestamps))
            
        # Calculate total processing time
        total_time = time.time() - start_time
        logging.info("Total speech analysis completed in %.2fs", total_time)
            
        # When in AI-only mode, we should prioritize AI timestamps
        if not self.keyword_trigger and self.use_ai_speech_analysis:
            # In AI-only mode, we return AI timestamps (not keywords)
            logging.info("AI-only mode: Using only AI-detected timestamps")
            return self.ai_timestamps
            
        # For standard mode, return keyword timestamps (AI timestamps are added separately in two_phase_process)
        return self.keyword_timestamps

    def detect_scene_changes_fast(self, video_processor, fps, frame_count, progress_callback=None):
        """
        Quickly scan through the video at a lower sampling rate to detect major scene changes.
        
        Args:
            video_processor: The VideoProcessor instance to get frames from
            fps: Frames per second of the video
            frame_count: Total number of frames
            progress_callback: Optional callback for progress updates
            
        Returns:
            list: List of timestamps where major scene changes were detected
        """
        # Reset the video processor to start from the beginning
        scene_change_timestamps = []
        
        # For very long videos (>10k frames), use a more aggressive sampling
        if frame_count > 10000:
            sample_rate = 60  # Sample every 60 frames (1 frame per second at 60fps)
        else:
            sample_rate = 30  # Sample every 30 frames
            
        logging.info("Fast scene change detection with sampling rate 1:%d", sample_rate)
        
        prev_frame = None
        prev_processed_frame = None
        
        frames_to_process = range(0, frame_count, sample_rate)
        total_frames_to_process = len(frames_to_process)
        
        for i, frame_number in enumerate(frames_to_process):
            # Update progress
            if progress_callback and i % 5 == 0:
                progress_callback(i / total_frames_to_process, 
                                 f"Scanning for scene changes: {i}/{total_frames_to_process} frames")
            
            # Get the frame
            frame = video_processor.get_frame_at_position(frame_number)
            if frame is None:
                continue
                
            timestamp = frame_number / fps
                
            # Skip the first frame
            if prev_frame is None:
                prev_frame = frame
                
                # Resize frame for processing if needed
                if frame.shape[0] > 720:
                    scale = 720 / frame.shape[0]
                    width = int(frame.shape[1] * scale)
                    prev_processed_frame = cv2.resize(frame, (width, 720))
                else:
                    prev_processed_frame = frame
                continue
                
            # Resize frame for processing if needed
            if frame.shape[0] > 720:
                scale = 720 / frame.shape[0]
                width = int(frame.shape[1] * scale)
                frame_resized = cv2.resize(frame, (width, 720))
            else:
                frame_resized = frame
                
            # Calculate structural similarity
            similarity = self.calculate_ssim(frame_resized, prev_processed_frame)
            
            # Low similarity means significant scene change - use a stricter threshold
            # for the fast scan to only catch major changes
            if similarity < self.ssim_threshold - 0.1:  # More strict threshold
                logging.info("Major scene change detected at %.2fs (SSIM=%.3f)", timestamp, similarity)
                
                # Add multiple timestamps around the scene change
                # This ensures we capture the frames right before and after the transition
                buffer_before = 0.5  # seconds before change
                buffer_after = 1.0   # seconds after change
                
                for t in [timestamp - buffer_before, timestamp, timestamp + buffer_after]:
                    if t >= 0:  # Ensure valid timestamp
                        self.scene_change_timestamps.append((t, f"Scene change (SSIM={similarity:.3f})"))
            
            # Update the previous frame
            prev_frame = frame
            prev_processed_frame = frame_resized
            
        logging.info("Found %d scene changes", len(self.scene_change_timestamps))
        return self.scene_change_timestamps
        
    def two_phase_process(self, video_processor, fps, frame_count, progress_callback=None, teams_llm_config = None):
        """
        Perform two-phase processing:
        1. First pass: Extract key timestamps (speech + scene changes)
        2. Second pass: Process video in parallel chunks
        
        Uses parallel processing if enabled to significantly improve performance.
        
        Args:
            video_processor: VideoProcessor instance
            fps: Frames per second of the video
            frame_count: Total number of frames
            progress_callback: Optional callback function for progress updates
            
        Returns:
            list: List of tuples (image, timestamp, reason) representing screenshots
        """
        # Import modules
        import concurrent.futures
        from PIL import Image
        
        # Print debug information about the video
        logging.info("===== STARTING VIDEO PROCESSING =====")
        logging.info("Video stats: %d frames at %.1f fps = %.2f seconds", frame_count, fps, frame_count/fps)
        
        start_time = time.time()
        
        # Track status for debugging
        phase1_complete = False
        phase2_started = False
        
        # Use parallel processing if enabled
        if self.parallel_processing:
            logging.info("Using TRUE PARALLEL processing for video chunks")
            
            # Phase 1: Run speech and scene detection sequentially to avoid import issues
            if progress_callback:
                progress_callback(0.0, "Processing...")
            
            logging.info("===== PHASE 1: SPEECH & SCENE DETECTION =====")
                
            try:
                # Run speech processing first
                logging.info("Starting speech processing...")
                speech_results = self.extract_all_speech_keywords(teams_llm_config=teams_llm_config)
                logging.info("Speech processing complete")
                
                # Run scene detection based on the UI setting
                # The 'use_ssim' property is set from app.py's checkbox
                if self.detection_mode=='advanced':
                    logging.info("Starting scene detection with structural similarity...")
                    scene_results = self.detect_scene_changes_fast(
                        video_processor, fps, frame_count,
                        lambda p, m: progress_callback(0.1 + p * 0.2, m) if progress_callback else None
                    )
                    logging.info("Scene detection complete")
                else:
                    logging.info("Scene detection skipped (disabled in settings)")
                    self.scene_change_timestamps = []  # Clear any existing scene changes
                
                # Mark Phase 1 as complete
                phase1_complete = True
                logging.info("Phase 1 complete - all detection methods finished")
            except Exception as e:
                logging.error("ERROR in Phase 1: %s", e)
                import traceback
                traceback.print_exc()
                
            if progress_callback:
                progress_callback(0.4, "Processing...")
            
            # Phase 2: Process video in parallel chunks using our parallel processor
            try:
                logging.info("\n===== PHASE 2: SCREENSHOT GENERATION =====")
                phase2_started = True
                logging.info("Phase 2 starting - preparing to generate screenshots")
                
                # Print debug information about what we have so far
                logging.info("\nDEBUG INFO for Phase 1 results:")
                logging.info(f"- Keyword timestamps: {len(self.keyword_timestamps)}")
                logging.info(f"- AI timestamps: {len(self.ai_timestamps)}")
                logging.info(f"- Scene changes: {len(self.scene_change_timestamps)}")
                
                # Make sure we have valid key timestamps first
                if not hasattr(self, 'key_timestamps') or not self.key_timestamps:
                    logging.info("\nWARNING: No key timestamps found for Phase 2")
                    logging.info("Creating combined timestamps from Phase 1 results")
                    
                    # Create timestamps from speech and scene detection results
                    self.key_timestamps = []
                    
                    # Add speech keyword timestamps
                    if self.keyword_timestamps:
                        logging.info(f"Adding {len(self.keyword_timestamps)} keyword timestamps")
                        self.key_timestamps.extend(self.keyword_timestamps)
                        
                    # Add AI timestamps
                    if self.ai_timestamps:
                        logging.info(f"Adding {len(self.ai_timestamps)} AI timestamps")
                        self.key_timestamps.extend(self.ai_timestamps)
                        
                    # Add scene changes
                    if self.scene_change_timestamps:
                        logging.info(f"Adding {len(self.scene_change_timestamps)} scene changes")
                        self.key_timestamps.extend(self.scene_change_timestamps)
                    
                    # If we still don't have timestamps, create basic ones
                    if not self.key_timestamps:
                        logging.info("No timestamps from Phase 1, creating basic timestamps")
                        video_duration = frame_count / fps
                        
                        # Add start, middle and end points
                        self.key_timestamps.append((0, "Start of video"))
                        self.key_timestamps.append((video_duration / 2, "Middle of video"))
                        self.key_timestamps.append((video_duration, "End of video"))
                
                # Sort timestamps and remove duplicates
                if self.key_timestamps:
                    logging.info(f"Sorting {len(self.key_timestamps)} timestamps")
                    self.key_timestamps.sort(key=lambda x: x[0])
                    
                    # Remove timestamps that are too close to each other
                    filtered_timestamps = []
                    last_time = -self.cooldown
                    
                    for timestamp, reason in self.key_timestamps:
                        if timestamp - last_time >= self.cooldown:
                            filtered_timestamps.append((timestamp, reason))
                            last_time = timestamp
                    
                    self.key_timestamps = filtered_timestamps
                    logging.info(f"After filtering: {len(self.key_timestamps)} timestamps remaining")
                else:
                    logging.info("ERROR: No timestamps available after Phase 1")
                    # Create some emergency timestamps
                    video_duration = frame_count / fps
                    num_samples = 5  # Take 5 screenshots evenly spaced
                    
                    for i in range(num_samples):
                        timestamp = (i * video_duration) / (num_samples - 1) if num_samples > 1 else 0
                        self.key_timestamps.append((timestamp, f"Emergency timestamp {i+1}"))
                
                # Use a simpler approach to avoid threading issues with video processing
                logging.info("\n--------------------------------")
                logging.info("STARTING PHASE 2: Sequential Video Processing")
                logging.info("--------------------------------")
                
                # Process timestamps sequentially to avoid frame access issues
                logging.info(f"Starting Phase 2: Processing {len(self.key_timestamps)} key timestamps")
                
                # Create a single list of screenshots
                screenshots = []
                face_blur_processor_fast = FastFaceBlurProcessor(blur_intensity=40)

                
                # Process each timestamp individually - more reliable than parallel
                for i, (timestamp, reason) in enumerate(self.key_timestamps):
                    logging.info(f"Processing timestamp {i+1}/{len(self.key_timestamps)}: {timestamp:.2f}s - {reason}")
                    
                    try:
                        # Convert timestamp to frame number
                        frame_number = int(timestamp * fps)
                        
                        # Add a frame position check to avoid out-of-range errors
                        if 0 <= frame_number < frame_count:
                            # Get the frame
                            frame = video_processor.get_frame_at_position(frame_number)
                            
                            if frame is not None:
                                try:
                                    blurred_frame = face_blur_processor_fast.blur_faces_in_frame(frame)
                                    img_pil = Image.fromarray(blurred_frame)
                                    screenshots.append((img_pil, timestamp, reason))
                                    logging.info(f"Added screenshot at {timestamp:.2f}s - {reason}")
                                except Exception as img_err:
                                    logging.exception(f"Error converting frame to image: {img_err}, trying alternate method.")

                                    try:
                                        # Convert to PIL image without blur as fallback
                                        img_pil = Image.fromarray(frame)
                                        screenshots.append((img_pil, timestamp, reason))
                                        logging.info(f"Added screenshot at {timestamp:.2f}s - {reason}")
                                    except Exception as img_err:
                                        logging.exception(f"Error converting frame to image: {img_err}")
                        else:
                            logging.info(f"Frame number {frame_number} out of range (0-{frame_count-1})")
                    except Exception as e:
                        logging.exception(f"Error processing timestamp {timestamp:.2f}s: {e}")
                        try:
                            img_pil = Image.fromarray(frame)
                            screenshots.append((img_pil, timestamp, reason))
                            logging.info(f"Added screenshot at {timestamp:.2f}s - {reason} (original, blur failed)")
                        except Exception as fallback_err:
                            logging.exception(f"Error converting frame to image: {fallback_err}")
                
                # If we still don't have enough screenshots, sample frames directly
                if len(screenshots) < 5:
                    logging.info("Not enough screenshots from timestamps, adding samples from video")
                    
                    # Take a few evenly spaced frames
                    total_duration = frame_count / fps
                    num_samples = min(5, int(total_duration / 30))  # At most 5 or one per 30 seconds
                    
                    for i in range(num_samples):
                        sample_time = (i * total_duration) / num_samples if num_samples > 0 else 0
                        frame_num = int(sample_time * fps)
                        
                        if 0 <= frame_num < frame_count:
                            try:
                                sample_frame = video_processor.get_frame_at_position(frame_num)
                                if sample_frame is not None:
                                    try:
                                        blurred_frame = face_blur_processor_fast.blur_faces_in_frame(sample_frame)
                                        img_pil = Image.fromarray(blurred_frame)
                                        screenshots.append((img_pil, timestamp, reason))
                                        logging.info(f"Added screenshot at {timestamp:.2f}s - {reason}")
                                    except Exception as img_err:
                                        logging.exception(f"Error converting frame to image: {img_err}, trying alternate method.")

                                        try:
                                            # Convert to PIL image without blur as fallback
                                            img_pil = Image.fromarray(sample_frame)
                                            screenshots.append((img_pil, timestamp, reason))
                                            logging.info(f"Added screenshot at {timestamp:.2f}s - {reason}")
                                        except Exception as img_err:
                                            logging.exception(f"Error converting frame to image: {img_err}")
                            except Exception as e:
                                logging.exception(f"Error processing timestamp {timestamp:.2f}s: {e}") 
                                try:
                                    img_pil = Image.fromarray(sample_frame)
                                    screenshots.append((img_pil, timestamp, reason))
                                    logging.info(f"Added screenshot at {timestamp:.2f}s - {reason} (original, blur failed)")
                                except Exception as fallback_err:
                                    logging.exception(f"Error converting frame to image: {fallback_err}")       
                # Sort screenshots by timestamp
                screenshots.sort(key=lambda x: x[1])
                
                logging.info(f"Phase 2 completed: Generated {len(screenshots)} screenshots")
            except Exception as e:
                logging.exception(f"Error in parallel video processing: {e}")
                import traceback
                traceback.print_exc()
                
                # Fall back to sequential processing
                logging.info("Falling back to sequential processing...")
                screenshots = []
                
                # Process each key timestamp sequentially
                if hasattr(self, 'key_timestamps') and self.key_timestamps:
                    logging.info(f"Processing {len(self.key_timestamps)} timestamps sequentially")
                    
                    for i, (timestamp, reason) in enumerate(self.key_timestamps):
                        # Convert timestamp to frame number
                        frame_number = int(timestamp * fps)
                        
                        # Get the frame
                        frame = video_processor.get_frame_at_position(frame_number)
                        if frame is not None:
                            # Convert to PIL image
                            img_pil = Image.fromarray(frame)
                            # Add to screenshots
                            screenshots.append((img_pil, timestamp, reason))
                            logging.info(f"Sequential processing: Screenshot at {timestamp:.2f}s - {reason}")
                
            # If we still don't have screenshots, create some basic ones
            if not screenshots:
                logging.info("ERROR: No screenshots generated. Creating emergency screenshots...")
                
                # We need to create basic screenshots to ensure we always have output
                emergency_screenshots = []
                
                # Sample evenly throughout the video
                video_duration = frame_count / fps
                num_samples = 5  # Take 5 screenshots evenly spaced
                
                for i in range(num_samples):
                    # Calculate timestamp
                    timestamp = (i * video_duration) / (num_samples - 1) if num_samples > 1 else 0
                    # Convert to frame number
                    frame_number = int(timestamp * fps)
                    
                    if frame_number < frame_count:
                        # Get the frame
                        frame = video_processor.get_frame_at_position(frame_number)
                        if frame is not None:
                            # Convert to PIL image
                            img_pil = Image.fromarray(frame)
                            # Add to screenshots
                            emergency_screenshots.append((img_pil, timestamp, f"Emergency screenshot {i+1}"))
                
                # Use these emergency screenshots
                screenshots = emergency_screenshots
                logging.info(f"Created {len(screenshots)} emergency screenshots")
                
            logging.info(f"Total screenshots generated: {len(screenshots)}")
            
            processing_time = time.time() - start_time
            logging.info(f"Parallel video processing complete: {len(screenshots)} screenshots in {processing_time:.2f}s")
            return screenshots
        
        # Use sequential processing otherwise
        screenshots = []
        logging.info("Using sequential processing")
        
        # Phase 1: Extract key timestamps
        if progress_callback:
            progress_callback(0.0, "Processing speech data...")
        
        # 1a. Process all speech to find keywords or AI triggers
        self.extract_all_speech_keywords(teams_llm_config=teams_llm_config)
        
        # 1b. Find major scene changes - ONLY if not in AI-only mode
        # Always detect scene changes regardless of AI settings
        if progress_callback:
            progress_callback(0.2, "Processing...")
        
        self.detect_scene_changes_fast(video_processor, fps, frame_count, 
                                     lambda p, m: progress_callback(0.2 + p * 0.3, m) if progress_callback else None)
                                     
        # Log the detection mode
        if self.use_ai_speech_analysis:
            if progress_callback:
                progress_callback(0.35, "Processing...")
        
        # Combine all timestamps and sort
        all_timestamps = []
        all_timestamps.extend(self.keyword_timestamps)
        
        # Add scene change timestamps
        if len(self.scene_change_timestamps) > 0:
            logging.info(f"Adding {len(self.scene_change_timestamps)} scene changes to process")
            all_timestamps.extend(self.scene_change_timestamps)
        else:
            logging.info("Warning: No scene changes detected or enabled")
        
        # Add AI-detected timestamps if they exist
        if self.use_ai_speech_analysis:
            all_timestamps.extend(self.ai_timestamps)
            
        # Add start and end frames
        all_timestamps.append((0, "Start of video"))
        all_timestamps.append((frame_count / fps, "End of video"))
        
        # Print all collected timestamps for debugging
        logging.info(f"All collected timestamps before filtering: {len(all_timestamps)}")
        for ts, reason in all_timestamps[:10]:  # Print first 10 for debugging
            logging.info(f"  - {ts:.2f}s: {reason}")
        if len(all_timestamps) > 10:
            logging.info(f"  ... and {len(all_timestamps) - 10} more")
        
        # Sort by timestamp and remove duplicates or too close timestamps
        all_timestamps.sort(key=lambda x: x[0])
        
        filtered_timestamps = []
        last_time = -self.cooldown
        
        for timestamp, reason in all_timestamps:
            # Only keep timestamps that are at least cooldown seconds apart
            if timestamp - last_time >= self.cooldown:
                filtered_timestamps.append((timestamp, reason))
                last_time = timestamp
                
        # Print filtered timestamps for debugging
        logging.info(f"After filtering: {len(filtered_timestamps)} timestamps to process")
        
        # Store the final list of timestamps to process
        self.key_timestamps = filtered_timestamps
        
        # Phase 2: Process only frames at key timestamps
        if progress_callback:
            progress_callback(0.5, f"Processing...")
            
        # Always use the comprehensive window-based processing method
        # to ensure we capture the best frames at each timestamp
        # For each key timestamp, process frames in a small window around it
        for i, (timestamp, reason) in enumerate(self.key_timestamps):
            if progress_callback:
                progress_callback(0.5 + (i / len(self.key_timestamps)) * 0.5, 
                               f"Processing timestamp {i+1}/{len(self.key_timestamps)}: {timestamp:.2f}s")
            
            # Convert timestamp to frame number
            center_frame_number = int(timestamp * fps)
            
            # Special handling for scene changes - use exact frame for scene changes
            if "Scene change" in reason:
                logging.info(f"Processing scene change at {timestamp:.2f}s")
                frame = video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    screenshots.append((img_pil, timestamp, reason))
                    # Mark as processed
                    self.processed_timestamps.add(center_frame_number)
                continue  # Skip normal processing for scene changes
            
            # Special handling for AI detection - capture exact frame
            elif "AI Detected" in reason:
                logging.info(f"Processing AI detection at {timestamp:.2f}s")
                # For AI detections, always capture the exact frame to ensure we have it
                frame = video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    img_pil = Image.fromarray(frame)
                    # Add to screenshots but still continue normal processing to find the best frame
                    ai_screenshots = (img_pil, timestamp, reason)
                    screenshots.append(ai_screenshots)
                    # Mark as processed
                    self.processed_timestamps.add(center_frame_number)
                
            # Process a small window around the timestamp to find the best frame
            window_size = 10  # frames to check on each side
            best_frame = None
            best_reason = reason
            
            # Reset frame buffer
            self.frame_buffer = []
            self.prev_frame = None
            
            # Check frames in the window
            for offset in range(-window_size, window_size + 1):
                frame_number = center_frame_number + offset
                
                # Skip if out of range
                if frame_number < 0 or frame_number >= frame_count:
                    continue
                    
                # Skip if we've already processed this frame number
                if frame_number in self.processed_timestamps:
                    continue
                    
                # Mark as processed
                self.processed_timestamps.add(frame_number)
                
                # Get the frame
                frame = video_processor.get_frame_at_position(frame_number)
                if frame is None:
                    continue
                    
                # Calculate exact timestamp
                exact_timestamp = frame_number / fps
                
                # Run regular key frame detection on this frame
                is_key_frame, frame_reason = self.is_key_frame(frame, exact_timestamp)
                
                if is_key_frame:
                    best_frame = frame
                    best_reason = frame_reason
                    break
                
            # If no key frame was found in the window, use the center frame
            if best_frame is None:
                frame = video_processor.get_frame_at_position(center_frame_number)
                if frame is not None:
                    best_frame = frame
            
            # Add the screenshot if we found a good frame
            if best_frame is not None:
                img_pil = Image.fromarray(best_frame)
                screenshots.append((img_pil, timestamp, best_reason))
        
        return screenshots

    def set_detection_mode(self, mode):
        """
        Set the detection mode for screenshot extraction.
        
        Args:
            mode (str): 'basic' for all methods except scene changes, 'advanced' for all methods including scene changes
        """
        if mode not in ['basic', 'advanced']:
            logging.warning("Invalid detection mode '%s'. Using 'advanced' as default.", mode)
            mode = 'advanced'
        
        self.detection_mode = mode
        logging.info("Detection mode set to: %s", mode)
        
        if mode == 'basic':
            logging.info("Basic mode: Keyword triggers + Mouse tracking + AI analysis (no scene changes)")
        else:
            logging.info("Advanced mode: All detection methods including scene change detection")

    def __del__(self):
        """Clean up temporary files."""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.unlink(self.temp_audio_file)
            except:
                pass

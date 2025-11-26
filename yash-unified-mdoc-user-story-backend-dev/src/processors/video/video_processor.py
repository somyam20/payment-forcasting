import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_path):
        """
        Initialize the video processor with the path to the video file.
        
        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
    def __del__(self):
        """Release the video capture when the object is destroyed."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            
    def get_frames(self):
        """
        Generator that yields frames from the video.
        
        Yields:
            tuple: (frame as numpy array, frame number)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB (Streamlit/PIL uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_rgb, frame_number
            frame_number += 1
            
    def get_frame_at_position(self, frame_number):
        """
        Get a specific frame from the video.
        
        Args:
            frame_number (int): The number of the frame to retrieve.
            
        Returns:
            numpy.ndarray: The frame as a numpy array, or None if the frame could not be retrieved.
        """
        if frame_number < 0 or frame_number >= self.frame_count:
            return None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

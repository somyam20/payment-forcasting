# Add this import at the top of your file
import cv2
# import face_recognition
import numpy as np
from PIL import Image

# Add this class near the top of your file or import it
# class FaceBlurProcessor:
#     def __init__(self, blur_intensity=15):
#         self.blur_intensity = blur_intensity
    
#     def blur_faces_in_frame(self, frame: np.ndarray) -> np.ndarray:
#         """
#         Blur faces in a single frame (numpy array).
        
#         Args:
#             frame: numpy array representing the image frame
        
#         Returns:
#             Frame with blurred faces
#         """
#         try:
#             # Convert BGR to RGB for face_recognition (if frame is BGR)
#             # Note: Check if your frames are RGB or BGR format
#             if len(frame.shape) == 3 and frame.shape[2] == 3:
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self._is_bgr_frame(frame) else frame
#             else:
#                 rgb_frame = frame
            
#             # Detect face locations
#             face_locations = face_recognition.face_locations(rgb_frame)
            
#             if not face_locations:
#                 return frame  # No faces found, return original
            
#             # Convert back to BGR for OpenCV operations if needed
#             work_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) if not self._is_bgr_frame(frame) else frame.copy()
            
#             # Blur each detected face
#             for (top, right, bottom, left) in face_locations:
#                 # Add padding around the face
#                 padding = int(min(right - left, bottom - top) * 0.1)
#                 x1 = max(0, left - padding)
#                 y1 = max(0, top - padding)
#                 x2 = min(work_frame.shape[1], right + padding)
#                 y2 = min(work_frame.shape[0], bottom + padding)
                
#                 # Extract and blur face region
#                 face_region = work_frame[y1:y2, x1:x2]
#                 blur_kernel_size = max(self.blur_intensity, 1)
#                 if blur_kernel_size % 2 == 0:
#                     blur_kernel_size += 1  # Kernel size must be odd
                
#                 blurred_face = cv2.GaussianBlur(face_region, (blur_kernel_size, blur_kernel_size), 0)
#                 work_frame[y1:y2, x1:x2] = blurred_face
            
#             return work_frame
            
#         except Exception as e:
#             # If face detection fails, return original frame
#             print(f"Face blur failed: {e}")
#             return frame
    
#     def _is_bgr_frame(self, frame):
#         """Simple heuristic to detect if frame is BGR (common in OpenCV)"""
#         # This is a basic check - you might need to adjust based on your specific case
#         return True  # Assume BGR by default, adjust if needed

# Alternative: Lightweight version using only OpenCV Haar cascades (faster but less accurate)
class FastFaceBlurProcessor:
    def __init__(self, blur_intensity=15):
        self.blur_intensity = blur_intensity
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def blur_faces_in_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            work_frame = frame.copy()
            for (x, y, w, h) in faces:
                # Add padding
                padding = int(min(w, h) * 0.2)
                x1, y1 = max(0, x - padding), max(0, y - padding)
                x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
                
                # Blur face region
                face_region = work_frame[y1:y2, x1:x2]
                blur_size = max(self.blur_intensity, 1)
                if blur_size % 2 == 0:
                    blur_size += 1
                blurred = cv2.GaussianBlur(face_region, (blur_size, blur_size), 0)
                work_frame[y1:y2, x1:x2] = blurred
            
            return work_frame
        except:
            return frame


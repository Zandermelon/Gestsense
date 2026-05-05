import mediapipe as mp
import cv2 as cv


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# the 21-point hand skeleton — pairs of indices to connect with lines
HAND_CONNECTIONS = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # palm
    (0, 17),
]

class Landmarker:
    def __init__(self, camera):
        self.model_path = r"C:\Users\Zander\Desktop\Code\Gestsense\hand_landmarker.task"
        
        self.options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        num_hands=2,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.on_result)

        self.camera = camera
        self.landmarker = HandLandmarker.create_from_options(self.options)

        self.latest_result = None
        
    
    def on_result(self, result, output_image, timestamp_ms):
         self.latest_result = result
         
         

    def process_frame(self):
            
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.camera.get_frame())
            self.landmarker.detect_async(image,self.camera.get_timestamp())
        
        
    def draw_landmarks(self, frame):
            if self.latest_result is None or not self.latest_result.hand_landmarks:
                return frame

            h, w = frame.shape[:2]
            for hand in self.latest_result.hand_landmarks:
                # landmarks are normalized 0–1, convert to pixel coords
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

                # skeleton lines
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

                # landmark dots
                for x, y in points:
                    cv.circle(frame, (x, y), 4, (0, 0, 255), -1)

            return frame


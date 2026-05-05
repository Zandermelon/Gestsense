import mediapipe as mp
from camera import Camera


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Landmarker:
    def __init__(self, camera):
        self.model_path = r"C:\Users\Zander\Desktop\Code\Gestsense\hand_landmarker.task"
        
        self.options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=self.model_path),
        num_hands=2,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.print_result)

        self.camera = camera
        
            
    # Create a hand landmarker instance with the live stream mode:
    def print_result(result, output_image, timestamp_ms):
        if result.hand_landmarks:
            print(f"[{timestamp_ms} ms] hands detected: {len(result.hand_landmarks)}")
        else:
            print(f"[{timestamp_ms} ms] no hands")

    def process_frame(self):
        with HandLandmarker.create_from_options(self.options) as landmarker:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.camera.get_frame())
            landmarker.detect_async(image,self.camera.get_timestamp())
        
        












    



    


    
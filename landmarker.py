import mediapipe as mp

model_path = r"C:\Users\Zander\Desktop\Code\Gestsense\hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        print(f"[{timestamp_ms} ms] hands detected: {len(result.hand_landmarks)}")
    else:
        print(f"[{timestamp_ms} ms] no hands")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:

    # The landmarker is initialized. Use it here.
    # ...
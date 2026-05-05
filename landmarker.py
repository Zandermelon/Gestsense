import mediapipe as mp
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class Landmarker:
    def __init__(self, model_path, num_hands=2):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=num_hands,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.latest_result = None

    def _on_result(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def process_frame(self, frame_bgr, timestamp_ms):
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.landmarker.detect_async(image, timestamp_ms)

    def draw_landmarks(self, frame):
        if self.latest_result is None or not self.latest_result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]
        for hand in self.latest_result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
            for x, y in points:
                cv.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame

    def get_hands(self, frame_width, frame_height):
        if self.latest_result is None or not self.latest_result.hand_landmarks:
            return []
        return [
            [(lm.x * frame_width, lm.y * frame_height, lm.z) for lm in hand]
            for hand in self.latest_result.hand_landmarks
        ]

    def close(self):
        self.landmarker.close()
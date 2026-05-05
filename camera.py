import time
import cv2 as cv


class Camera:
        def __init__(self, video=0):
                self.video = cv.VideoCapture(video)
                if not self.video.isOpened():
                        raise Exception("Couldn't open camera")
                self.start_ns = time.monotonic_ns()
                
        # Captures video input
        def get_frame(self):
                ret, frame = self.video.read()
                if not ret:
                        return None
                return cv.cvtColor(frame, cv.COLOR_BGR2RGB) # Convert frame to standard RGB
        
        def release(self):
                self.video.release()

        def get_timestamp(self):
                return (time.monotonic_ns() - self.start_ns) // 1000000 # convert nanoseconds to ms

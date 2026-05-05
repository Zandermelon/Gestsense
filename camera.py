import numpy as np
import cv2 as cv


class Camera:
        def __init__(self, video=0):
                self.video = cv.VideoCapture(video)
                if not self.video.isOpened():
                        raise Exception("Couldn't open camera")
                
        # Captures video input
        def get_frame(self):
                ret, frame = self.video.read()
                if not ret:
                        return None
                return frame
        
        def release(self):
                self.video.release()

        def get_timestamp(self):
                return self.video.get(cv.CAP_PROP_POS_MSEC)

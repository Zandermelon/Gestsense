import time
import cv2 as cv


class Camera:
    def __init__(self, video=0, width=640, height=480):
        self.video = cv.VideoCapture(video, cv.CAP_DSHOW)
        if not self.video.isOpened():
            raise RuntimeError("Couldn't open camera")

        self.video.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.video.set(cv.CAP_PROP_BUFFERSIZE, 1)

        self.start_ns = time.monotonic_ns()

    def get_frame(self):
        ret, frame = self.video.read()
        return frame if ret else None

    def get_timestamp(self):
        return (time.monotonic_ns() - self.start_ns) // 1_000_000

    def release(self):
        self.video.release()
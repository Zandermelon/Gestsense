import cv2 as cv
from camera import Camera
from landmarker import Landmarker

MODEL_PATH = r"C:\Users\Zander\Desktop\Code\Gestsense\hand_landmarker.task"


def main():
    cam = Camera(0)
    landmarker = Landmarker(MODEL_PATH)

    try:
        while True:
            frame = cam.get_frame()
            if frame is None:
                continue

            landmarker.process_frame(frame, cam.get_timestamp())
            landmarker.draw_landmarks(frame)

            cv.imshow("Gestsense", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        landmarker.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
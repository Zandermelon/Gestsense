import cv2 as cv
from camera import Camera
from landmarker import Landmarker

cam = Camera(0)
landmarker = Landmarker(cam)




try:
    while True:
        frame = cam.get_frame()

        if landmarker.camera.get_frame() is None:
            continue

        landmarker.process_frame()
        frame = landmarker.draw_landmarks(landmarker.camera.get_frame())


        cv.imshow("Gestsense", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cam.release()
    cv.destroyAllWindows()







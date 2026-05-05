import cv2 as cv
from camera import Camera
from landmarker import Landmarker

cam = Camera(0)
landmarker = Landmarker(cam)

landmarker.print_result

while True:
    frame = cam.get_frame()

    if frame is None:
        break

    cv.imshow("Gestsense", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv.destroyAllWindows()




import cv2 as cv
from camera import Camera

cam = Camera(0)

while True:
    frame = cam.get_frame()

    if frame is None:
        break

    cv.imshow("Gestsense", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv.destroyAllWindows()




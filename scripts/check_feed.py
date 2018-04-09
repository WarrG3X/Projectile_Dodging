import cv2
import os



c = cv2.VideoCapture(1)
c.set(6,1196444237.0)
width,height = c.get(3),c.get(4)
print("Frame Width/Height : ", width, height)
print("FPS : ",c.get(5))

while True:
    _,f = c.read()
    cv2.imshow('Output',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
c.release()

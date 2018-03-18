import cv2
import os
import time


def updateFocus(focus):
    os.system("v4l2-ctl -d 1 --set-ctrl=focus_absolute="+str(focus))

def updateExposure(exp):
    os.system("v4l2-ctl -d 1 --set-ctrl=exposure_absolute="+str(exp))


cv2.namedWindow('Output')
cv2.createTrackbar('Focus','Output',1,1023, updateFocus)
cv2.createTrackbar('Exposure','Output',1,5000, updateExposure)
os.system("v4l2-ctl -d 1 --set-ctrl=focus_auto=0")
os.system("v4l2-ctl -d 1 --set-ctrl=exposure_auto=1")
time.sleep(1)
cv2.setTrackbarPos('Focus','Output',451)
cv2.setTrackbarPos('Exposure','Output',482)
os.system("v4l2-ctl -d 1 --set-ctrl=focus_absolute=451")
os.system("v4l2-ctl -d 1 --set-ctrl=exposure_absolute=482")



c = cv2.VideoCapture(1)
c.set(6,1196444237.0)
width,height = c.get(3),c.get(4)
print "Frame Width/Height : ", width, height
print "FPS : ",c.get(5)

while True:
    _,f = c.read()
    cv2.imshow('Output',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
c.release()

import cv2
import os
import time
import numpy as np
import yaml


def load_values():
    with open("config.yaml","r") as file:
        data = yaml.load(file)["param"]
        values = [data['y'],data['u'],data['v'],data['Y'],data['U'],
                data['V'],data['erode_iter'],data['dil_iter']]
        return values
    

c = cv2.VideoCapture(1)
c.set(6,1196444237.0)
width,height = c.get(3),c.get(4)
print "Frame Width/Height : ", width, height
print "FPS : ",c.get(5)

while True:
    _,f = c.read()
    yuv = cv2.cvtColor(f,cv2.COLOR_BGR2YUV)
    y,u,v,Y,U,V,erode_iter,dil_iter, = load_values()
    
    mask = cv2.inRange(yuv,np.array([y,u,v]),np.array([Y,U,V]))
    mask = cv2.erode(mask,None,iterations=erode_iter)
    mask = cv2.dilate(mask,None,iterations=dil_iter)
    res = cv2.bitwise_and(f,f,mask=mask)

    img,contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)!=0:
        cnt = max(contours,key=cv2.contourArea)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        centre,radius = (int(x),int(y)),int(radius)
        cv2.circle(f,centre,radius,(0,255,0),2)


    cv2.imshow('Console',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
c.release()

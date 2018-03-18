import cv2
import os
import time
import numpy as np
import yaml


def get_console_values():
    value_list = [cv2.getTrackbarPos("Lower Y","Console"),
                  cv2.getTrackbarPos("Lower U","Console"),
                  cv2.getTrackbarPos("Lower V","Console"),
                  cv2.getTrackbarPos("Higher Y","Console"),
                  cv2.getTrackbarPos("Higher U","Console"),
                  cv2.getTrackbarPos("Higher V","Console"),
                  cv2.getTrackbarPos("Erode","Console"),
                  cv2.getTrackbarPos("Dilate","Console"),
                  cv2.getTrackbarPos("Mode","Console"),]
    return value_list


def load_console_values():
    try:
        with open("config.yaml","r") as file:
            data = yaml.load(file)["param"]
            cv2.setTrackbarPos("Lower Y","Console",data['y'])
            cv2.setTrackbarPos("Lower U","Console",data['u'])
            cv2.setTrackbarPos("Lower V","Console",data['v'])
            cv2.setTrackbarPos("Higher Y","Console",data['Y'])
            cv2.setTrackbarPos("Higher U","Console",data['U'])
            cv2.setTrackbarPos("Higher V","Console",data['V'])
            cv2.setTrackbarPos("Erode","Console",data['erode_iter'])
            cv2.setTrackbarPos("Dilate","Console",data['dil_iter'])
            cv2.setTrackbarPos("Mode","Console",0)
            print values

    except:
        pass
    

def write_console_values():

    dictkeys = ['y','u','v','Y','U','V','erode_iter','dil_iter']
    dictvalues = get_console_values()[:-1]
    config_dict = dict(zip(dictkeys,dictvalues))
    param_dict = {"param" : config_dict}
    with open("config.yaml",'w') as file:
        yaml.dump(param_dict,file,default_flow_style=False)
        print "Config Updated"

cv2.namedWindow('Console')
cv2.createTrackbar('Lower Y','Console',0,255,lambda x:x)
cv2.createTrackbar('Lower U','Console',0,255,lambda x:x)
cv2.createTrackbar('Lower V','Console',0,255,lambda x:x)
cv2.createTrackbar('Higher Y','Console',0,255,lambda x:x)
cv2.createTrackbar('Higher U','Console',0,255,lambda x:x)
cv2.createTrackbar('Higher V','Console',0,255,lambda x:x)
cv2.createTrackbar('Erode','Console',0,30,lambda x:x)
cv2.createTrackbar('Dilate','Console',0,30,lambda x:x)
cv2.createTrackbar('Mode','Console',0,2,lambda x:x)


c = cv2.VideoCapture(1)
c.set(6,1196444237.0)
width,height = c.get(3),c.get(4)
print "Frame Width/Height : ", width, height
print "FPS : ",c.get(5)
load_console_values()

while True:
    _,f = c.read()
    yuv = cv2.cvtColor(f,cv2.COLOR_BGR2YUV)
    y,u,v,Y,U,V,erode_iter,dil_iter,mode = get_console_values()
    
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


    if mode == 0:
        cv2.imshow('Console',f)
    elif mode == 1:
        cv2.imshow('Console',mask)
    else:
        cv2.imshow('Console',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

write_console_values()
cv2.destroyAllWindows()
c.release()

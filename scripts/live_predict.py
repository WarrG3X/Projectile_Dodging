import cv2
import os
import time
import numpy as np
import yaml

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


fid = 0
status = 0
N_FRAMES = 7
IMG_BUFFER = []
X_BUFFER = []
Y_BUFFER = []
R_BUFFER = []



def load_values():
    with open("config.yaml","r") as file:
        data = yaml.load(file)["param"]
        values = [data['y'],data['u'],data['v'],data['Y'],data['U'],
                data['V'],data['erode_iter'],data['dil_iter']]
        return values
    

def detect(f):
    global fid
    global status

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
        cv2.circle(res,centre,radius,(0,255,0),2)
        cv2.circle(f,centre,radius,(0,255,0),2)
        X_BUFFER.append(round(x,3))
        Y_BUFFER.append(round(y,3))
        R_BUFFER.append(radius)
        IMG_BUFFER.append(res)
        fid += 1

        if(status)<0:
            status = 0
        status +=1
        return True


    if(status) > 0:
        status = 0
    status -=1
    return False

    

c = cv2.VideoCapture(1)
c.set(6,1196444237.0)
width,height = c.get(3),c.get(4)
print("Frame Width/Height : ", width, height)
print("FPS : ",c.get(5))

started = False

while True:
    _,f = c.read()
    
    bool_detect = detect(f)
    print(bool_detect,status)

    if status > 3 and not started:
        started = True

    if status < - 10 and started:
        break
    

    cv2.imshow('Console',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Frames = ",fid)

if fid < N_FRAMES:
    print("Insufficient Frames")
    exit()

X_BUFFER =  X_BUFFER[-N_FRAMES:]
Y_BUFFER =  Y_BUFFER[-N_FRAMES:]
R_BUFFER =  R_BUFFER[-N_FRAMES:]
IMG_BUFFER = IMG_BUFFER[-N_FRAMES:]

#Preview
idx = 0
while True:
    cv2.imshow('Console',IMG_BUFFER[idx])
    time.sleep(0.1)
    idx += 1
    if idx == N_FRAMES:
        idx = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cv2.destroyAllWindows()
c.release()


label_dict ={1:"lefthand",2:"righthand",3:"leftleg",4:"rightleg",
        5:"torso",6:"miss",7:"cancel"}

X_vec = []
[X_vec.extend(x) for x in zip(X_BUFFER,Y_BUFFER,R_BUFFER)]
X_vec = np.array(X_vec,dtype=np.float32).reshape(1,21)

print(X_vec)

class Model(torch.nn.Module):

    def __init__(self):

        super().__init__()
        self.layer1 = torch.nn.Linear(21,17)
        self.layer2 = torch.nn.Linear(17,13)
        self.layer3 = torch.nn.Linear(13,9)
        self.layer4 = torch.nn.Linear(9,5)



    def forward(self,x):
        
        out1 = F.relu(self.layer1(x))
        out2 = F.relu(self.layer2(out1))
        out3 = F.relu(self.layer3(out2))
        y_pred = F.relu(self.layer4(out3))

        return F.softmax(y_pred)



model = Model()
model.load_state_dict(torch.load('trainedmodel'))
X = Variable(torch.from_numpy(X_vec))

pred_vec = model(X)
print(pred_vec.data)
class_pred = torch.max(pred_vec,1)[1]
class_pred = label_dict[int(class_pred.data)+1]
print(class_pred)






import numpy as np
import cv2
import os
import random



def get_images(path):
    dir_list = os.listdir(path)
    dir_list = list(set([int(x[:-6])for x in  dir_list]))
    imgs = []
    name =  random.choice(dir_list)
    for i in range(1,8):
        imgs.append(path+str(name)+"-"+str(i)+".jpg")

    return imgs



LH = "image_db/lefthand/"
RH = "image_db/righthand/"
LL = "image_db/leftleg/"
RL = "image_db/rightleg/"
T = "image_db/torso/"

labels = [LH,RH,LL,RL,T]

imgs  = {label:get_images(label) for label in labels}

for label in labels:
    cv2.namedWindow(label)





i = 0

while True:

    for label in labels:
        f = cv2.imread(imgs[label][i])
        cv2.imshow(label,f)
    i+=1
    if i==7:
        imgs  = {label:get_images(label) for label in labels}
        i=0

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

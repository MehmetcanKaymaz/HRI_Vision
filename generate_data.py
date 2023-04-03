import cv2
import os

camera = camera = cv2.VideoCapture(0)


n_data=200
n_class=5

for i in range(n_class):
    input(f"Press enter to collect data for class {i+1}")
    path="data/test/class"+str(i+1)+"/"
    if os.path.exists(path)==False:
        os.mkdir(path)

    for j in range(n_data):
        _,image= camera.read()
        cv2.imwrite(path+"image"+str(j)+".png",image)
    
    
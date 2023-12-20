import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('digit_trained.h5')
run = False
ix,iy = -1,-1
follow = 25
img = np.zeros((512,512,1))


def draw(event, x, y, flag, params):
    global run,ix,iy,img,follow
    if event == cv2.EVENT_LBUTTONDOWN:
        run = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if run == True:
            cv2.circle(img, (x,y), 20, (255,255,255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        run = False
        cv2.circle(img, (x,y), 20, (255,255,255), -1)
        gray = cv2.resize(img, (28, 28))
        gray = gray.reshape(1, 784)
        result = np.argmax(model.predict(gray))
        result = 'Recognise : {}'.format(result)
        cv2.putText(img, org=(25,follow), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, text= result, color=(255,0,0), thickness=1)
        follow += 25
    elif event == cv2.EVENT_RBUTTONDOWN:
        img = np.zeros((512,512,1))
        follow = 25


### param
cv2.namedWindow('farah project')
cv2.setMouseCallback('farah project', draw)

while True:    
    cv2.imshow("farah project", img)
   
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()




import cv2
import numpy as np
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To ignore AVX2 FMA extensions

model = load_model('trained_model.h5')

# range for color detection
lower = np.array([50,50,50])
upper = np.array([130,255,255])
font = cv2.FONT_HERSHEY_SIMPLEX
ans = -1
map = {5:0,6:7,9:2,3:3,1:4,2:5,7:6,0:1,8:8,4:9} #in accordance with one-hot output

pts = []
image = np.zeros((480, 640, 3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,10.0,(640,480))

def get_contours(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    return cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def if_drawing(contours):
    contours = max(contours,key = cv2.contourArea) # find the largest contour
    # print(np.asarray(contours).shape)
    M = cv2.moments(np.asarray(contours)) #find the COM of this contour
    # print(M)
    if M['m00']: # corresponds to area
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        pts.append(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        cv2.line(image, pts[i - 1], pts[i], (255, 255, 255), 7) # 10 is thickness
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)


def done_drawing():
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.medianBlur(image_gray, 15)
    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
    _,thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_contours = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
    cv2.drawContours(image,image_contours, -1, (255,255,255), 3)
    cv2.imshow('image',image)

    if len(image_contours) >= 1:
            best = max(image_contours, key=cv2.contourArea)
            # print(cv2.contourArea(cnt))
            if cv2.contourArea(best) >= 1000:
                x, y, w, h = cv2.boundingRect(best)
                digit = image_gray[y:y + h, x:x + w]
                sample = cv2.resize(digit, (28, 28))
                cv2.imwrite('img.png',sample)
                sample = sample/255.0
                sample = sample.reshape(28,28,1)
                sample = np.asarray([sample]) # changing dimension to 1 X 28 X 28 X 1
                output = model.predict(sample)
                indices = np.where(output == output.max())
                # print(output)
                # print(indices,"yo")
                return indices[1][0]



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    _, contours ,_ = get_contours(frame)


    if len(contours)>=1:
        if_drawing(contours)

    else:
        if len(pts):
            ans = done_drawing()

        #re-initialize for next prediction
        pts.clear()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        digit = np.zeros((200, 200, 3), dtype=np.uint8)

    if ans != -1 and ans != None:
        cv2.putText(frame,"Prediction: "+str(map[ans]),(0,130),font,1,(255,0,0),2,cv2.LINE_AA)

    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

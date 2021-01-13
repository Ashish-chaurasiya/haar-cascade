# Haar Cascade Object Detection Face & Eye OpenCV Python Tutorial
### In this OpenCV with Python tutorial, we're going to discuss object detection with Haar Cascades. We'll do face and eye detection to start. In order to do object recognition/detection with cascade files, you first need cascade files. For the extremely popular tasks, these already exist. Detecting things like faces, cars, smiles, eyes, and license plates for example are all pretty prevalent.

### First, I will show you how to use these cascade files, then I will show you how to embark on creating your very own cascades, so that you can detect any object you want, which is pretty darn cool!

### You can use Google to find various Haar Cascades of things you may want to detect. You shouldn't have too much trouble finding the aforementioned types. We will use a Face cascade and Eye cascade. You can find a few more at the root directory of Haar cascades. Note the license for using/distributing these Haar Cascades.

### Let's begin our code. I am assuming you have downloaded the haarcascade_eye.xml and haarcascade_frontalface_default.xml from the links above, and have these files in your project's directory.

```python
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()




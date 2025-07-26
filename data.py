import cv2
import os

#load data
faces=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")

if not os.path.exists("dataset"):
    os.makedirs("dataset")

#ask for id and name
user_id=input("Enter Numeric ID :")
name=input("Enter your Name :")

#start camera

cap=cv2.VideoCapture(0)
count=0
print("Sample Capturing")

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=faces.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in face:
        count+=1
        face_img= gray[y:y+h,x:x+w]
        file=f"dataset/User.{user_id}.{count}.jpg"
        cv2.imwrite(file,face_img)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,f"Sample {count}/30",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.imshow("capture",frame)
    if cv2.waitKey(1) == 27 or count >=30:
        break

cap.release()
cv2.destroyAllWindows()
print("completed")

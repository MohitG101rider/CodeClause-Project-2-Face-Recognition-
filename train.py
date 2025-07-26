import cv2
import numpy as np
import os

dataset_path="data"
trainer_path="trainer"

#LBPH recognizer
recog=cv2.face.LBPHFaceRecognizer_create()

faces=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

sample_face=[]
face_id=[]
print("loading images")
for file in os.listdir(dataset_path):
    if file.endswith((".jpg",".jpeg",".png")):
        path=os.path.join(dataset_path,file)
        gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        if gray is None:
            print(f"Skipping file {file}")
            continue
        gray=cv2.resize(gray,(200,200))

        try:
            id=int(file.split(".")[1])
        except:
            print(f"bad fromat file {file}")
            continue
        sample_face.append(gray)
        face_id.append(id)

print(f" total faces:{len(sample_face)}")

if len(sample_face)<2:
    print("not enough faces to train")
else:
    recog.train(sample_face,np.array(face_id))
    if not os.path.exists(trainer_path):
        os.makedirs(trainer_path)

recog.save("trainer/trained_model.yml")
print("training completed")

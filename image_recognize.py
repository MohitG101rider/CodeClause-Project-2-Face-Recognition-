import cv2
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trained_model.yml")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Map IDs to names (same as in training)
names = {
    5: "Narendra Modi",
    6: "Narendra Modi",
    7: "Narendra Modi",
    8: "Narendra Modi"
}

# Load the test image
print(" test images you can enter -\n 1. test9.jpg\n 2. test10.jpg")
img_path = input("Enter test image filename: ").strip()
print('Done')
image = cv2.imread(img_path)

if image is None:
    print("Image not found .")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (300,200))

    id_, confidence = recognizer.predict(face_resized)

    if confidence < 70:
        name = names.get(id_, "Unknown")
        label = f"{name} ({round(100 - confidence+50)}%)"
    else:
        label = "Unknown"

    # Draw box and label
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

cv2.imshow("Image Face Recognition", image)
print(f"Confidence = {label} sure")
cv2.waitKey(0)
cv2.destroyAllWindows()

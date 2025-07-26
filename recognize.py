import cv2
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trained_model.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# You can map user IDs to names here
names = {
    5: "Narendra Modi",
    6: "Narendra Modi",
    7: "Narendra Modi",
    8: "Narendra Modi"
}

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        # predict face ID and confidence
        id_, confidence = recognizer.predict(cv2.resize(face_img, (200, 200)))

        if confidence < 70:
            name = names.get(id_, "Unknown")
            label = f"{name} ({round(100 - confidence+50)}%)"
        else:
            label = "Unknown"

        # label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # exit
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
print("Webcam closed")

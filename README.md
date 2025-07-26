# CodeClause-Project-2-Face-Recognition-
CodeClause Internship

#Face Recognition Project using Opencv LBPH (new try because i am unable to use face recognition in my laptop due to error in .whl building)

(A) If you want to test my model -
    
 1. Open all files in your system via VS code ,idle or any other compiler.
 2. Keep all the files in one folder.
 3. I have trained my model on "Narendra Modi".
 4. Run *recognize.py* to open webcam and show any picture of "Narendra Modi" to the camera , it will recognize it with confidence meter.
 5. Run *imange_recognize.py* , it ask for image name - you can use my uploaded image (test9.jpg, test10.jpg) or you can also use any other image [.jpg,.png,.jpeg only] but don't            forget to put it in folder where all files are.
 6. Note - LBPH is not as accurate as face-recognition library , but i have tried my best to make it as much accurate as face-recognition.

(B) If you want to train data with your own faces -
    1. Firstly,make a folder named "data" and run *data.py* .
    2. It will ask for ID and Name [ Id should be unique (eg-1,2,3,4,5 etc), but if you want to train more pic of the same person then, id can be different but the name should be same.
    3. Webcam will start and click the picture - you can train your own face or you can also open any person pic in you phone and place it in front of camera when it is capturing .
    4. It will save the 30 copies of the captured pic in the data folder.
    5. Then , run *train.py* , it will train the model with the pictures.
    6. To Test- (a) run *recognize.py* to test from webcam.
                (b) run *image_recognize* to test from image you have.

  - That's all , Thanks for viewing .

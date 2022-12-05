from datetime import datetime
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import json
time = datetime.now()
cap = cv2.VideoCapture(0)
# students = {'ziyad': []}

while (True):
    f = open("data.json", "r+")

    time = datetime.now()

    students = json.loads(f.read())

    # Capture frame by frame
    ret, img = cap.read()
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(img, 1.3, 5)
    rgb_small_frame = small_frame[:, :, ::-1]
    faces = face_recognition.face_locations(rgb_small_frame)

    if faces:
        cv2.imwrite("faces/1.jpg", img)     # save frame as JPEG file
        f1 = "faces/1.jpg"
        for i, j in students.items():

            f2 = f"known_faces/{i}.jpg"
            backends = ['opencv', 'ssd', 'dlib',
                        'mtcnn', 'retinaface', 'mediapipe']
            result = DeepFace.verify(img1_path=f1, img2_path=f2,
                                     detector_backend=backends[1], enforce_detection=False)
            print(result)
            if (result['verified'] == True):
                if (students.get(i, None) is None or type(students.get(i, None)) is not list):
                    students[i] = [[], [], 'Absent']
                if time.minute not in students[i][0]:
                    print(time.minute)
                    students[i][0].append(time.minute)
                    print(students)
                    students[i][1] = len(students[i][0])
                    if (len(students[i][0])) > 48:
                        students[i][2] = "Attendant"

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(students)

    f.seek(0)
    f.truncate()
    json.dump(students, f)
    f.close()

cap.release()

cv2.destroyAllWindows()

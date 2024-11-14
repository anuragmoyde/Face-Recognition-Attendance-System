from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

# Ensure Attendance directory exists
if not os.path.exists("Attendance"):
    os.makedirs("Attendance")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    attendance = []  # Initialize attendance only when face is detected

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        # Get timestamp for the attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        csv_path = f"Attendance/Attendance_{date}.csv"
        
        # Draw rectangles and add text for identified person
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # Save the name and timestamp to the attendance list
        attendance = [str(output[0]), str(timestamp)]

    # Resize frame and update background
    frame_resized = cv2.resize(frame, (640, 480))
    imgBackground[162:162 + 480, 55:55 + 640] = frame_resized
    cv2.imshow("Frame", imgBackground)
    
    # Capture key press
    k = cv2.waitKey(1)
    if k == ord('o') and attendance:  # Check that attendance data is available
        # Check if the CSV file exists, if not, create with headers
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)  # Write headers if file doesn't exist
            writer.writerow(attendance)  # Write the attendance record
        print(f"Attendance marked for {attendance[0]} at {attendance[1]}")

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

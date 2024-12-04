#!/usr/bin/env python3
import cv2
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
import re
import os

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

class FaceRecognition:
    def __init__(self):
        # Initialize HOG for human detection
        self.model = YOLO("/home/mustar/catkin_ws/src/cr_receptionist/models/yolov8n.pt")
        self.face_model = YOLO("/home/mustar/catkin_ws/src/cr_receptionist/models/yolov8n-face.pt")

        # Load the Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier('/home/mustar/catkin_ws/src/cr_receptionist/models/haarcascade_frontalface_default.xml')

        # Load the reference for deep face
        self.reference_db = "/home/mustar/catkin_ws/src/cr_receptionist/src/images"

    def detect_human(self, image):
        results = self.model.predict(source=image, classes=[0])

        if results:
            num = 0
            for objects in results:
                for result in objects:
                    a = result.boxes.cpu().numpy()
                    x1, y1, x2, y2 = map(int, a.xyxy[0])
                    num += 1
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            return num

        return False

    # def detect_faces(self, image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #     faces = self.face_cascade.detectMultiScale(
    #         gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    #     )

    #     # Draw rectangles around faces
    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #     return faces
    
    def detect_faces(self, image):
        results = self.face_model(image)
        faces = []

        # Draw bounding boxes around detected faces
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                faces.append((x1, y1, x2 - x1, y2 - y1))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return faces
    
    def recognize_faces(self, image):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Attempt to find faces in the frame and match them with reference images
        results = DeepFace.find(rgb_frame, self.reference_db, enforce_detection=False, silent=True)

        # Print results for debugging
        print("Recognition results:", results)

        # Check if any results were found
        if results and len(results) > 0:
            for df in results:
                if not df.empty:  # Check if the DataFrame is not empty
                    for index, row in df.iterrows():
                        name = f'{row.identity}'.split('/')[-1]
                        exact_name = re.match(r"([a-zA-Z]+)\d+", name)
                        
                        if exact_name:
                            print(f'face recognition done: {exact_name}')
                            # Debugging: Display detected face and name on the image
                            face_x, face_y, face_w, face_h = row.source_x, row.source_y, row.source_w, row.source_h
                            cv2.putText(image, f'{exact_name.group(1)}', (int(face_x) - 50, int(face_y) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
                            cv2.rectangle(image, (int(face_x), int(face_y)), (int(face_x + face_w), int(face_y + face_h)), BLUE, 3)
                            
                            # Return the recognized name
                            return exact_name.group(1)
                        
                else:
                    print("No matching faces found in the reference database.")
        else:
            print("No faces found in the frame or reference images missing.")

        # If no faces are recognized, return None
        return None

        # key = cv2.waitKey(1) & 0xFF
        # if key in (ord('q'), 27):  # 27 is the ESC key
        #     cv2.destroyAllWindows()

    def save_faces(self, image, person_name):
        print(f'Name: {person_name}')
        person_folder = os.path.join(self.reference_db, person_name)
        os.makedirs(person_folder, exist_ok=True)

        # Determine the filename for the new image: name1.jpg, name2.jpg, etc.
        image_count = len(os.listdir(person_folder)) + 1
        image_filename = f"{person_name}{image_count}.jpg"
        image_path = os.path.join(person_folder, image_filename)

        # Save the image to the created path
        cv2.imwrite(image_path, image)
        print(f"Saved image to {image_path}")

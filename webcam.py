import face_recognition_api
import cv2
import os
import pickle
import numpy as np
import warnings

video_capture = cv2.VideoCapture(0)

# Load classifier
fname = 'classifier.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        (le, clf) = pickle.load(f)
else:
    print('\x1b[0;37;43m' + f"Classifier '{fname}' does not exist" + '\x1b[0m')
    quit()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        print(f"[DEBUG] dtype: {small_frame.dtype}, shape: {small_frame.shape}, ndim: {small_frame.ndim}")

        if process_this_frame:
            face_locations = face_recognition_api.face_locations(small_frame)
            face_encodings = face_recognition_api.face_encodings(small_frame, face_locations)
            print(f"[DEBUG] Detected {len(face_encodings)} face(s).")

            predictions = []

            if len(face_encodings) > 0:
                closest_distances = clf.kneighbors(face_encodings, n_neighbors=1)
                is_recognized = [closest_distances[0][i][0] <= 0.5 for i in range(len(face_locations))]

                for encoding, location, recognized in zip(face_encodings, face_locations, is_recognized):
                    if recognized:
                        pred = clf.predict([encoding])[0]
                        name = le.inverse_transform([int(pred)])[0].title()
                        label = f"{name} - Authorized"
                        color = (0, 255, 0)  # Green
                        print(f"[DEBUG] Recognized: {label}")
                    else:
                        label = "Unauthorized"
                        color = (0, 0, 255)  # Red
                        print("[DEBUG] Face NOT recognized: Unauthorized")

                    predictions.append((label, location, color))

        process_this_frame = not process_this_frame

        # Draw results
        for label, (top, right, bottom, left), color in predictions:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.9, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

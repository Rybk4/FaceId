import cv2
import face_recognition
import os

# Function to load images and their encodings from a directory
def load_images_from_directory(directory):
    known_faces = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names

# Directory containing known faces
known_faces_directory = "known_faces"

# Load known faces and their encodings
known_faces, known_names = load_images_from_directory(known_faces_directory)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the current face matches any known person
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if any(matches):
            matched_name = known_names[matches.index(True)]
            name = f"Known: {matched_name}"

            # Print a message to the console when a known face is detected
            print(f"Detected {matched_name}!")

        # Draw rectangle around the face and display the result
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
video_capture.release()
cv2.destroyAllWindows()

import cv2
import pymongo
import io
import requests
import numpy as np
import face_recognition
import subprocess  # Import the subprocess module for opening images

# Connect to your MongoDB database
client = pymongo.MongoClient("mongodb+srv://user:user@cluster0.a87ri9o.mongodb.net/?retryWrites=true&w=majority")
db = client["test"]
collection = db["lawyers"]

def get_user_barcouncil_no(face_encoding):
    # Query the MongoDB collection to retrieve user data
    user_data = collection.find()

    for user in user_data:
        try:
            encoded_image = np.frombuffer(io.BytesIO(requests.get(user["profileImage"]).content).read(), np.uint8)
            face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(io.BytesIO(encoded_image)))

            if len(face_encodings) > 0:
                user_face_encoding = face_encodings[0]

                # Compare the face encodings
                if face_recognition.compare_faces([user_face_encoding], face_encoding)[0]:
                    return user["BarcouncilNO"]
        except Exception as e:
            # Handle the exception (e.g., log the error)
            print(f"Error loading image for user {user['BarcouncilNO']}: {str(e)}")
    return None

def capture_image_from_webcam():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    success, img = cap.read()

    # Release the webcam
    cap.release()

    return img

def main():
    # Capture an image from the webcam
    webcam_image = capture_image_from_webcam()

    # Perform face recognition on the captured image
    face_locations = face_recognition.face_locations(webcam_image)
    if not face_locations:
        print("No faces detected in the webcam-captured image.")
        return

    # Select the first detected face
    face_encoding = face_recognition.face_encodings(webcam_image)[0]

    # Find the corresponding Barcouncil number in the database
    barcouncil_no = get_user_barcouncil_no(face_encoding)

    if barcouncil_no is not None:
        print(f"Barcouncil number: {barcouncil_no}")

    # Draw rectangle around the detected face and add Barcouncil number
    top, right, bottom, left = face_locations[0]
    cv2.rectangle(webcam_image, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
    cv2.putText(webcam_image, f"BC: {barcouncil_no}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image to a file
    cv2.imwrite("webcam_image.jpg", webcam_image)

    # Open the saved image using the default image viewer on Windows
    try:
        subprocess.Popen(["start", "webcam_image.jpg"], shell=True)
    except Exception as e:
        print(f"Error opening image: {str(e)}")

if __name__ == "__main__":
    main()


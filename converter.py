from flask import Flask, request, send_file
from flask_pymongo import PyMongo
import cv2
import face_recognition
import io
import requests
import numpy as np
from bson import ObjectId
from io import BytesIO

app = Flask(__name__)
app.config['MONGO_URI'] = 'mongodb://collection:Aditya@#0006@localhost:27017/yourdb'
mongo = PyMongo(app)

# Route for uploading an image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        # Store the image in MongoDB
        mongo.db.images.insert_one({'image_data': image.read()})
        return 'Image uploaded successfully'
    return 'Image upload failed'

# Route for retrieving an image
@app.route('/image/<image_id>', methods=['GET'])
def get_image(image_id):
    image_data = mongo.db.images.find_one({'_id': ObjectId(image_id)})
    if image_data:
        return send_file(BytesIO(image_data['image_data']),mimetype='image/jpeg')  # Adjust mimetype as needed
    return 'Image not found', 404

# Route for face recognition
@app.route('/recognize/<barcouncil_no>', methods=['GET'])
def recognize_face(barcouncil_no):
    user_data = mongo.db.lawyers.find_one({"BarcouncilNO": barcouncil_no})

    if user_data is None:
        return 'User not found in the database', 404

    # Load the image from the URL stored in the database
    image_url = user_data["profileImage"]
    response = requests.get(image_url)

    if response.status_code != 200:
        return 'Failed to retrieve user image from the URL', 500

    image_bytes = io.BytesIO(response.content)
    image_array = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(imgS)
        encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces([encode_face], user_data.get("encode", []))
            face_dis = face_recognition.face_distance([encode_face], user_data.get("encode", []))

            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = user_data["BarcouncilNO"]
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    app.run(debug=True)
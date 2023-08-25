from flask import Flask, request, jsonify, render_template
import numpy as np
import datetime
import os
import cv2
import time
from tensorflow import lite
import queue
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)
app.debug = True

answer = -1
classes = {-1: "default", 0: "0_background", 1: "1_trash", 2: "2_paper", 3: "3_plastic", 4: "4_metal",
           5: "5_electronic_invoice", 6: "6_bubble_wrap", 7: "7_thin_plastic_bag", 8: "8_fruit_mesh_bag", 9: "9_thin_film_paper_cup"}

rpi_ip = "http://192.168.0.130:5555"

def predictions_verify(predictions):
    if not predictions.full():
        return False

    first = predictions.get()
    same_count = 1
    for i in range(2):
        if predictions.queue[i] == first:
            same_count += 1

    if same_count == 3:
        return True

    return False

def preprocess_image(raw_image, corners= np.array([[5, 538], [100, 120], [800, 120], [950, 538]], dtype=np.float32), size= 224):
    
    # TODO: adjust parameters
    raw_image = cv2.resize(raw_image,(960,540))

    target_corners = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)    
    matrix = cv2.getPerspectiveTransform(corners, target_corners)
    img = cv2.warpPerspective(raw_image, matrix, (size, size))
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, (size, size))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    upload_path = os.path.join(current_dir, 'static', 'uploads', "latest.jpg")
    cv2.imwrite(upload_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    # print('Input image shape:', x.shape)
    return x
    

# def Preprocess(img):
    
#     img = cv2.resize(img, (224, 224))

#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     upload_path = os.path.join(current_dir, 'static', 'uploads', "latest.jpg")

#     cv2.imwrite(upload_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
#     x = np.expand_dims(img, axis=0)
#     x = preprocess_input(x)
#     # print('Input image shape:', x.shape)
#     return x


@app.route('/home')
def home():
    timestamp = int(time.time())
    return render_template('home.html', timestamp=timestamp, prediction=classes[prediction_id], answer=classes[answer])


@app.route('/finish_place', methods=['GET'])  # with Frontend
def finish_place():

    rpi_server_url = rpi_ip + "/captured_image"
    global answer
    start = datetime.datetime.now()
    answer = get_rpi_result(rpi_server_url)
    print("Update answer:", classes[answer])
    end = datetime.datetime.now()
    print("Inference time: ",(end-start))
    
    # TODO: special case check
    if classes[answer] == "9_thin_film_paper_cup":
        print("\nspecial case: 9_thin_film_paper_cup\n")
        data = {'text_data': 'T'}
        rpi_server_url = rpi_ip + "/arduino_signal"
        response = requests.post(rpi_server_url, data=data)
        print("Rpi response: ",response.content)

    trash_classes = {
        1: "Trash",
        2: "Paper",
        3: "Plastic",
        4: "Metal",
        5: "ElectronicInvoice",
        6: "BubbleWrap",
        7: "ThinPlasticBag",
        8: "FruitMeshBag",
        9: "ThinFilmPaperCup",
    }
    return jsonify({'answer': trash_classes[answer]})


@app.route('/answer', methods=['GET'])  # with Frontend
def get_answer():
    args = request.args

    global answer
    print("Answer:", answer)
    print("User select:", args.get("select"))  # "right" or "left"

    correct = False
    if args.get("select") == "right":  # recycle
        if answer == 2 or answer == 3 or answer == 4 or answer == 8 or answer == 7:
            correct = True
        else:
            correct = False

    elif args.get("select") == "left":  # trash
        if answer == 1 or answer == 5 or answer == 6:
            correct = True
        else:
            correct = False

    trash_classes = {
        1: "Trash",
        2: "Paper",
        3: "Plastic",
        4: "Metal",
        5: "ElectronicInvoice",
        6: "BubbleWrap",
        7: "ThinPlasticBag",
        8: "FruitMeshBag",
        9: "ThinFilmPaperCup",
    }
    print("is_correct:",correct)
    print("answer:",trash_classes[answer])


    # send signal to Rpi to drive the motor
    if answer == 2 or answer == 3 or answer == 4 or answer == 8 or answer == 7:
        data = {'text_data': 'R'}
    else:
        data = {'text_data': 'L'}

    rpi_server_url = rpi_ip + "/arduino_signal"
    response = requests.post(rpi_server_url, data=data)
    print("Rpi response: ",response.content)

    return jsonify({'is_correct': correct, 'answer': trash_classes[answer]})


def get_rpi_result(rpi_server_url):

    count = 0
    prediction_id = -1

    # TODO: check queue maxsize
    predictions_queue = queue.Queue(maxsize=3)

    while not predictions_verify(predictions_queue):
        response = requests.get(rpi_server_url, stream=True)
        if response.status_code == 200:
            bytes_data = b''
            for chunk in response.iter_content(chunk_size=1024):
                bytes_data += chunk
            image = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow('Received Image', image)
            # print(type(image))
            # print(image.shape)
            # time.sleep(1)

            if image is None:
                print("Failed to load the image")

            prediction_id = predict_class(image)
            predictions_queue.put(prediction_id)
            count+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to receive image from rpi")
            break
    
    return prediction_id

def predict_class(input_image):

    input_data = preprocess_image(input_image)
    TFLite_interpreter.set_tensor(input_details[0]['index'], input_data)
    TFLite_interpreter.invoke()
    output_data = TFLite_interpreter.get_tensor(output_details[0]['index'])

    global prediction_id
    prediction_id = np.argmax(output_data)
    print("\nPrediction: {}".format(classes[prediction_id]))

    return prediction_id


if __name__ == '__main__':

    # TODO: revise EfficientNet model
    tflite_model_path = os.path.join(os.path.abspath(os.getcwd()), "EfficientNetB0_V6.tflite")
    TFLite_interpreter = lite.Interpreter(model_path=tflite_model_path)
    TFLite_interpreter.allocate_tensors()
    input_details = TFLite_interpreter.get_input_details()
    output_details = TFLite_interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    app.run(host='0.0.0.0')



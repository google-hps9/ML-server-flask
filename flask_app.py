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

app = Flask(__name__)
app.debug = True

prediction_id = -1
answer = -1
classes = {-1:"default",0:"0_background",1:"1_trash",2:"2_paper"
           ,3:"3_plastic",4:"4_metal",5:"5_electronic_invoice",6:"6_bubble_wrap"
           ,7:"7_thin_plastic_bag",8:"8_fruit_mesh_bag",9:"9_thin_film_paper_cup"}

predictions_queue = queue.Queue(maxsize=5)

def predictions_verify(predictions=predictions_queue):
    if not predictions_queue.full():
        return False
    
    first = predictions_queue.get()
    same_count = 1
    for i in range(4):
        if predictions_queue.queue[i] == first:
            same_count+=1

    if same_count==5:
        return True

    return False

def Preprocess(img):
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    upload_path = os.path.join(current_dir, 'static', 'uploads', "latest.jpg")

    cv2.imwrite(upload_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    # print('Input image shape:', x.shape)
    return x

@app.route('/home')
def home():
    timestamp = int(time.time())
    return render_template('home.html', timestamp=timestamp, prediction=classes[prediction_id], answer=classes[answer])


@app.route('/answer',methods=['GET']) # with Frontend
def get_answer():
    args = request.args
    print("User select:",args.get("select")) # "right" or "left" 
    ans_reported = classes[answer]
    print("Answer:",ans_reported)

    correct = False

    if args.get("select") == "right": # trash
        if ans_reported == "not yet":
            print("No answer yet")
        if ans_reported == "jji" or "ff":
            correct = True
            print("Correct")
        if ans_reported == "jji" or "ff":
            correct = False
            print("Wrong")

    elif args.get("select") == "left": # recycle
        if ans_reported == "not yet":
            print("No answer yet")
        if ans_reported == "jji" or "ff":
            correct = True
            print("Correct")
        if ans_reported == "jji" or "ff":
            correct = False
            print("Wrong")


    return jsonify({'is_correct': correct,'answer': ans_reported})


@app.route('/predict',methods=['POST']) # with RPi
def process_image():
    
    input_image = request.files['image'].read()

    start = datetime.datetime.now()
    input_data = Preprocess(input_image)
    TFLite_interpreter.set_tensor(input_details[0]['index'], input_data)
    TFLite_interpreter.invoke()
    output_data = TFLite_interpreter.get_tensor(output_details[0]['index'])

    global prediction_id
    prediction_id = np.argmax(output_data)
    print("\nPrediction: {}".format(classes[prediction_id]))

    predictions_queue.put(prediction_id)
    if predictions_verify(predictions_queue):  # if last 5 predictions are the same      
        global answer
        answer = prediction_id
        print("Update answer:",classes[answer])

    end = datetime.datetime.now()
    # print("Inference time: {} ms".format(int((end-start).microseconds/1000)))
    
    return jsonify({'result': str(prediction_id)})


if __name__ == '__main__':

    # Load the TFLite model
    tflite_model_path  = os.path.abspath(os.getcwd())+"\\EfficientNetB0_V6.tflite"
    TFLite_interpreter = lite.Interpreter(model_path=tflite_model_path)
    TFLite_interpreter.allocate_tensors()
    input_details = TFLite_interpreter.get_input_details()
    output_details = TFLite_interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    app.run(host='0.0.0.0')
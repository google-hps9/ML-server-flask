from flask import Flask, request,Response, jsonify, render_template
import numpy as np
import datetime
import os
import cv2
import threading
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
import requests # for http request
import serial # for communication with Arduino
from stream_stability import check_stream_stability

# declare
server_url = 'http://192.168.0.164:5000/predict' 
arduino = '/dev/ttyUSB0'
app = Flask(__name__)
app.debug = True

cam = cv2.VideoCapture(0) # setup cam
frame_lock = threading.Lock()


@app.route('/captured_image', methods=['POST'])
def getCapturedImage():
    # check stability
    global cam
    with frame_lock:
        ret, frame = cam.read()
        if ret:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
            
    return 'Failed to capture image', 500


    # proceed, pic = imageProcess(cam=cam)
    # if(proceed):
    #     result = httpRequest(frame=pic)
    
    # print("predict result:",result)


@app.route('/arduino_signal', methods=['POST'])
def sendArduinoSignal():
    
    result = request.files['motor'].read()

    # communicate with Arduino
    if(result == 'L'):
        finish = signal('L')
    elif(result == 'R'):
        finish = signal('R')

    print("arduino signal has been sent:",finish)


def imageCapture():
    # image capture
    ret, frame = cam.read()
    if(ret):
        return frame

def imageProcess(cam):
    # image Process
    # if object exists and to be classified, return true
    isStable, frame = check_stream_stability(cam, MOTION_THRESHOLD=5000)
    return isStable, frame

def signal(c):
    # send signal to arduino
    #!/usr/bin/env python3
    ser = serial.Serial(arduino, 9600, timeout=1)
    ser.reset_input_buffer()
    while True:
        ser.write(c.encode('utf-8'))
        line = ser.readline().decode('utf-8').rstrip()
        if(line == 'F'):
            return True
        else: continue

def httpRequest(frame):
    # send image and request classification result
    _, f = cv2.imencode('.jpg', frame)  # convert cv2 to jpg
    files = {'image': ('filename.jpg', f, 'image/jpeg')}
    start = datetime.datetime.now()
    response = requests.post(server_url, files=files) # request for result
    end = datetime.datetime.now()
    # print("Response time: {} ms".format(int((end-start).microseconds/1000)))

    

    if response.status_code == 200:
        result = response.json()['result']
        return result
    else:
        return "NULL"
    

if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    app.run(host='0.0.0.0')

    # while(True):
    #     # check stability
    #     proceed, pic = imageProcess(cam=cam)
    #     if(proceed):
    #         result = httpRequest(frame=pic)
    #     else:
    #         continue

    #     print("predict result:",result)

    #     # communicate with Arduino
    #     if(result == 'L'):
    #         finish = signal('L')
    #     elif(result == 'R'):
    #         finish = signal('R')
    #     else:
    #         continue
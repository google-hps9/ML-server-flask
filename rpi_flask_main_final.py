from flask import Flask, request,Response, jsonify, render_template
import numpy as np
import datetime
import os
import cv2
import threading
import requests # for http request
import serial # for communication with Arduino
from stream_stability import check_stream_stability
import time
import smbus # for I2C
# sudo apt-get install i2c-tools
# sudo apt-get install python-smbus

# declare 
arduino = '/dev/ttyUSB0'
app = Flask(__name__)
app.debug = True
bus = smbus.SMBus(0)
address = 0x60 # i2cdetect -y 0 to see the address
ser = serial.Serial(arduino, 9600, timeout=1)

frame_lock = threading.Lock()
#cam = cv2.VideoCapture(0)

def capture_frames():
    global cam
    while True:
        ret, frame = cam.read()
        if ret:
            with frame_lock:
                current_frame = frame
        time.sleep(0.1)

@app.route('/captured_image')
def getCapturedImage():
    
    # check stability
    
    cam = cv2.VideoCapture(0)
    while frame_lock:
        ret, frame = cam.read()
        print(ret)
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
        # finish = signalI2C('L')
    elif(result == 'R'):
        finish = signal('R')
        # finish = signalI2C('R')

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
    ser.reset_input_buffer()
    # time.sleep(5)
    ser.write(c.encode('utf-8'))
    while True:
        line = ser.readline().decode('utf-8').rstrip()
        if(line == 'F'):
            return True
        else: continue

def signalI2C(c):
    bus.write_byte_data(address, 0, c)
    while True:
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
    app.run(host='0.0.0.0',port=5555)

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
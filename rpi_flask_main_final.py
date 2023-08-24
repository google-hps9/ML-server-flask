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

def capture_frames():
    cam = cv2.VideoCapture(0)
    while True:
        ret, frame = cam.read()
        if ret:
            with frame_lock:
                current_frame = frame
        time.sleep(0.1)

@app.route('/captured_image')
def getCapturedImage():
    cam = cv2.VideoCapture(0)

    # check stability
    while True:
        isStable, frame = check_stream_stability(cam, MOTION_THRESHOLD=5000)
        if isStable:    
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
    

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
    

if __name__ == "__main__":
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    app.run(host='0.0.0.0',port=5555)

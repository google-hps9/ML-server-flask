from flask import Flask, Response, request
import cv2
import threading
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)
frame_lock = threading.Lock()

def capture_frames():
    global camera
    while True:
        ret, frame = camera.read()
        if ret:
            with frame_lock:
                current_frame = frame
        time.sleep(0.1)

@app.route('/get_image')
def get_image():
    global camera
    with frame_lock:
        ret, frame = camera.read()
        if ret:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
    return 'Failed to capture image', 500

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    app.run(host='0.0.0.0', port=5000)

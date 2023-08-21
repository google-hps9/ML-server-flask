import requests
import datetime
import cv2

server_url = 'http://172.20.10.3:5000/predict'  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("captured image",frame)

    _, image = cv2.imencode('.jpg', frame)
    files = {'image': ('filename.jpg', image, 'image/jpeg')}
    response = requests.post(server_url, files=files)

    if response.status_code == 200:
        json_response = response.json()
        result = json_response['result']
        print("Prediction:",result)
    else:
        print('Failed to send image to the server.')


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

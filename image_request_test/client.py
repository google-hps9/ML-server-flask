import requests
import cv2
import numpy as np
import time

url = 'http://192.168.0.168:5555/captured_image'  # 替换为树莓派的IP地址

while True:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        bytes_data = b''
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
        image = cv2.imdecode(np.frombuffer(bytes_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('Received Image', image)
        time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("无法获取影像")
        break

cv2.destroyAllWindows()

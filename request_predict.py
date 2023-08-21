import requests
import os
import datetime

server_url = 'http://172.20.10.3:5000/process_image'  

with open('C:\\Users\\ASUS\\Desktop\\HPS\\flask_server\\samples\\paper582.jpg', 'rb') as file:
    image_data = file.read()

files = {'image': ('filename.jpg', image_data, 'image/jpeg')}

start = datetime.datetime.now()
response = requests.post(server_url, files=files)
end = datetime.datetime.now()
print("Response time: {} ms".format(int((end-start).microseconds/1000)))

print(response)
if response.status_code == 200:
    result = response.json()['result']
    print(result)
else:
    print('Failed to send image to the server.')


from flask import Flask, render_template, Response
import cv2
import numpy as np
import youtube_dl
import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx

app = Flask(__name__)

# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
# Compile the model for faster speed
net.hybridize()

# camera = cv2.VideoCapture(-1)
# time.sleep(1) ### letting the camera autofocus

video_url = 'https://www.youtube.com/watch?v=LzwXgwVpa7Q'

ydl_opts = {}

# create youtube-dl object
ydl = youtube_dl.YoutubeDL(ydl_opts)

# set video url, extract video information
info_dict = ydl.extract_info(video_url, download=False)

# get video formats available
formats = info_dict.get('formats',None) 

def gen_frames(): 
    for f in formats:

        # I want the lowest resolution, so I set resolution as 144p
        if f.get('format_note',None) == '240p':

            #get the video url
            url = f.get('url',None)

            # open url with opencv
            cap = cv2.VideoCapture(url)

            # check if url was opened
            if not cap.isOpened():
                print('video not opened')
                exit(-1)

            while True:
                # read frame
                ret, frame = cap.read()

                # check if frame is empty
                if not ret:
                    break

                # Image pre-processing
                frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
                rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

                # Run frame through network
                class_IDs, scores, bounding_boxes = net(rgb_nd)

                # Display the result
                img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)

                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
    
 
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
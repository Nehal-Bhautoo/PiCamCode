# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
import serial


# sudo rfcomm listen /dev/rfcomm1

app = Flask(__name__, template_folder="templates")
# try:
#     bluetoothSerial = serial.Serial("/dev/rfcomm1", baudrate=115200)
# except:
#     print("error")


def get_frame():
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size = (640, 480))
    avg = None

    # allow the camera to adjust to lighting/white balance
    time.sleep(2)

    # initiate video or frame capture sequence
    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw array representation of the image
        frame = f.array
        
        # convert imags to grayscale &  blur the result
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # inittialize avg if it hasn't been done
        if avg is None:
            avg = gray.copy().astype("float")
            rawCapture.truncate(0)
            continue
        
        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, avg, 0.05)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        # coonvert the difference into binary & dilate the result to fill in small holes
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours or continuous white blobs in the image
        contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # find the index of the largest contour
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]   

            # draw a bounding box/rectangle around the largest contour
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            area = cv2.contourArea(cnt)

            # add text to the frame
            cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        # show the frame
        imageCode = cv2.imencode('.jpg', frame)[1]
        #print(frame.dtype)
        stringData = imageCode.tostring()
#         get_blueCon(imageCode)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + stringData + b'\r\n\r\n') 
        




# def get_blueCon(frame):
#     try:
#         bluetoothSerial.write(frame)
#     except SerialException:
#         print("QUIT")


    


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/vid')
def vid():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    



if __name__ == '__main__':
    app.run(host='192.168.100.27', port=5200, debug=False, threaded=True)



cv2.destroyAllWindows()

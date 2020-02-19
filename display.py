from flask import Flask, Response, render_template, make_response
import cv2, time, threading
from dbconnect import connection
c, conn = connection()

print(c.execute("SELECT * FROM example"))
# sql = "INSERT INTO example (id,data) VALUES (%s,%s)"
# val = [(6,"hi"),(7,"hi")]
# c.executemany(sql, val)
# conn.commit()
# exit()
# c.close()
# conn.close()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/registrasi')
def registrasi():
    return 'halaman registrasi'

@app.route('/log')
def log():
    return 'halaman log'

@app.route('/list')
def list():
    recording()
    return 'halaman list'

def generate():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,img = cap.read()
        if(ret == True):
            img = cv2.flip(img,1)
        #     img = cv2.resize(img,(480,360))
            frame = cv2.imencode(".jpg", img)[1].tobytes()
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #     time.sleep(0.1)
        else:
            break

@app.route('/video_feed')
def video_feed():
    return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(debug=True)
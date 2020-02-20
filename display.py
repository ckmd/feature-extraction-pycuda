from flask import Flask, session, Response, redirect, render_template, make_response, request, url_for, escape
import cv2, time, threading, os, sys
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
cap = cv2.VideoCapture(0).release()
app.secret_key = '123'

@app.route('/')
def home():
    if 'username' in session:
      username = session['username']
      return render_template("index.html", ses = escape(username))
    return "You are not set current user <br><a href = '/login'></b>" + \
    "click here to log in</b></a>"
# app.add_url_rule('/','/home',home)

@app.route('/login', methods = ['GET', 'POST'])
def login():
   if request.method == 'POST':
      session['username'] = request.form['username']
      return redirect(url_for('home'))
   if 'username' in session:
      return redirect(url_for('home'))
   return render_template("login.html")

@app.route('/logout')
def logout():
    # remove the username from the session if it's there
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/registrasi')
def registrasi():
    return render_template("registrasi.html", ses = escape(session['username']))

@app.route('/log')
def log():
    return 'halaman log'

@app.route('/user')
def user():
    return 'All Registerede User'

# untuk melihat detail tiap user
@app.route('/user/<int:userID>')
def userdetail(userID):
    # return 'hi user id : %d ' % userID
    return render_template("detailUser.html", uid = userID)

def generate():
    global cap
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret,img = cap.read()
        if(ret == True):
            img = cv2.flip(img,1)
            frame = cv2.imencode(".jpg", img)[1].tobytes()
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
    

@app.route('/video_feed')
def video_feed():
    return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/registrasi_stream')
def registrasi_stream():
    return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/capture')
def capture():
    global cap
    print(cap)
    if(cap.isOpened()):
        ret,img = cap.read()
        img = cv2.flip(img,1)
        cv2.imwrite(filename='saved_img.jpg', img=img)
        cap.release()
        print(session['username'])
    return 'capture'

if __name__ == '__main__':
    app.run(debug=True)
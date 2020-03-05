from flask import Flask, session, Response, redirect, render_template, make_response, request, url_for, escape
import cv2, time, threading, os, sys, numpy, dlib, Gabor, json
from dbconnect import connection
from flask_bootstrap import Bootstrap
from datetime import datetime
# import pycuda.autoinit
import pycuda.driver as cuda

# kafka needed
from datetime import datetime
from pykafka import KafkaClient

# client = KafkaClient(hosts="localhost:9092")
# topic = client.topics['match5']
# producer = topic.get_sync_producer()
# kafka needed

data = {}
data['jetsonid'] = '001'
data['features'] = {}
def stream(features):
  data['timestamp'] = str(datetime.now().replace(microsecond=0))
  for dol in range(len(features)):
      data['features'][dol+1] = features[dol+1]
  message = json.dumps(data)
  producer.produce(message.encode('ascii'))

c, conn = connection()
rec = datetime.now().replace(microsecond=0)
# print(c.execute("SELECT * FROM example"))
# sql = "INSERT INTO log (id,user_id,jetson_id, recorded_time) VALUES (%s,%s,%s,%s)"
# sql = "INSERT INTO jetson (id,location) VALUES (1,'gate 1')"
# val = (1,1,1,rec)
# c.executemany(sql, val)
# c.execute(sql)
# conn.commit()
# c.close()
# conn.close()
# exit()
app = Flask(__name__)
# Bootstrap(app)
cap = cv2.VideoCapture(0).release()
app.secret_key = '123'
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Rachmad_ws/python/shape_predictor_68_face_landmarks.dat")
print("Hellocuda")
x,y = 480,360

# Gabor filter
f33r = Gabor.filter1long.astype(numpy.float32)
f33i = Gabor.filter1ilong.astype(numpy.float32)
f17r = Gabor.filter2long.astype(numpy.float32)
f17i = Gabor.filter2ilong.astype(numpy.float32)
f9r = Gabor.filter3long.astype(numpy.float32)
f9i = Gabor.filter3ilong.astype(numpy.float32)
f5r = Gabor.filter4long.astype(numpy.float32)
f5i = Gabor.filter4ilong.astype(numpy.float32)
# Result variables Declaration
r33r = numpy.zeros_like(f33r)
r33i = numpy.zeros_like(f33i)
r17r = numpy.zeros_like(f17r)
r17i = numpy.zeros_like(f17i)
r9r = numpy.zeros_like(f9r)
r9i = numpy.zeros_like(f9i)
r5r = numpy.zeros_like(f5r)
r5i = numpy.zeros_like(f5i)

@app.route('/')
def home():
    global cap
    if(cap is not None):
      cap.release()
    if 'username' in session:
      username = session['username']
      return render_template("index.html", ses = escape(username))
    return render_template("login.html")
#     "You are not set current user <br><a href = '/login'></b>" + \
#     "click here to log in</b></a>"
# # app.add_url_rule('/','/home',home)

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
    if 'username' in session:
      username = session['username']
      if 'sesicapture' in session:
        return render_template("registrasi.html", ses = escape(username), sesi = escape(session['sesicapture']))
      else:
        return render_template("registrasi.html", ses = escape(username), sesi = 1)
    return render_template("login.html")

@app.route('/log')
def log():
    global cap
    if(cap is not None):
      cap.release()
    # original query
    # c.execute("SELECT l.id, u.nama, j.location, l.recorded_time FROM log l INNER JOIN user u ON l.user_id = u.id INNER JOIN jetson j ON l.jetson_id = j.id")
    c.execute("SELECT l.id, l.user_id, j.location, l.recorded_time FROM log l INNER JOIN jetson j ON l.jetson_id = j.id")
    logget = c.fetchall()
    totlog = c.execute("SELECT * FROM log")
    return render_template("log.html", logs = logget, totallog = totlog)

@app.route('/recognizing')
def recognizing():
    return render_template("recognizing.html")

@app.route('/user')
def user():
    global cap
    if(cap is not None):
      cap.release()
    c.execute("SELECT id, nama, nrp FROM user")
    userget = c.fetchall()
    totuser = c.execute("SELECT * FROM user")
    return render_template("user.html", users = userget, totaluser = totuser)

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
    

@app.route('/matching_stream')
def matching_stream():
    global cap
    if(cap is None):
      print("cap is gabut")
    else:
      cap.release()
    return Response(matching(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/registrasi_stream')
def registrasi_stream():
    global cap
    if(cap is None):
      print("cap is gabut")
    else:
      cap.release()
    return Response(generate(),mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/capture/<sesi>/<pose>')
def capture(pose,sesi):
    # tambahkan session di sini
    code1 = 0
    code2 = 0
    session['sesicapture'] = sesi
    if(int(sesi) == 1):
      code1 = 'A'
      code2 = 'B'
    elif(int(sesi) == 2):
      code1 = 'C'
      code2 = 'D'
    elif(int(sesi) == 3):
      code1 = 'E'
      code2 = 'F'

    global cap
    usname = escape(session['username'])
    if(cap.isOpened()):
        ret,img = cap.read()
        ori = img
        img = cv2.flip(img,1)
        if not os.path.exists('/home/er2c-jetson-nano/registered/'+usname):
            print("buat folder "+usname)
            os.makedirs('registered/'+usname)
        cv2.imwrite(filename='registered/'+usname+'/'+usname+'_'+code1+'_'+pose+'.jpg', img=img)
        if(int(pose) < 0):
          strpos = '+'+str(int(pose)*-1)
        else:
          strpos = str(int(pose)*-1)
        cv2.imwrite(filename='registered/'+usname+'/'+usname+'_'+code2+'_'+strpos+'.jpg', img=ori)
        cap.release()
    return 'capture'

def hitungcuda(a5):
    # cuda.init()
    # device = cuda.Device(0)
    # ctx = device.make_context()
    from pycuda.compiler import SourceModule
    mod = SourceModule("""
    __global__ void conv5(float *r5r, float *r5i, float *a5, float *f5r, float *f5i)
    {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        int Idx = i + j * blockDim.x * gridDim.x;
        r5r[Idx] = a5[Idx] * f5r[Idx];
        r5i[Idx] = a5[Idx] * f5i[Idx];
    }
    """)
    # ctx.pop()
    conv5 = mod.get_function("conv5")
    conv5(cuda.Out(r5r), cuda.Out(r5i), cuda.In(a5), cuda.In(f5r), cuda.In(f5i), block=(68,4,1), grid=(5,5))
    # return

def matching():
  global cap
  cap = cv2.VideoCapture(0)
  while(cap.isOpened()):
      ret,cam = cap.read()
      if(ret == True):
        cam = cv2.flip(cam,1)
        frame = cv2.resize(cam,(x,y))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # numpy.random.seed(1)
    # for s in range(len(subject90)):
        start = time.time()
    #     ri = numpy.random.randint(len(subject90))
    #     frame = subject90[ri]
    #     grey = frame
        faces = face_detector(grey)
        for face in faces:
            all_area33 = []
            all_area17 = []
            all_area9 = []
            all_area5 = []
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            landmark = predictor(grey, face)
            if(x1 > 20 and y1 > 20 and x2 < (x-20) and (y2 < y-20)):
              # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),4)
              for i in range(0,68):
                  x0 = landmark.part(i).x
                  y0 = landmark.part(i).y
                  cv2.circle(frame, (x0,y0), 1, (0,255,255),2)

                  # get sorround area of each single point
                  area33 = grey[y0-16:y0+17, x0-16:x0+17]/255.0
                  area17 = grey[y0-8:y0+9, x0-8:x0+9]/255.0
                  area9 = grey[y0-4:y0+5, x0-4:x0+5]/255.0
                  area5 = grey[y0-2:y0+3, x0-2:x0+3]/255.0

                  # multiply each sorround area 4 times for same with filter for 4 orientation convolution
                  for dup in range(4):
                      all_area33.append(area33)
                      all_area17.append(area17)
                      all_area9.append(area9)
                      all_area5.append(area5)

              # flattening before enter cuda calculation
              a33 = numpy.concatenate(all_area33).ravel().astype(numpy.float32)
              a17 = numpy.concatenate(all_area17).ravel().astype(numpy.float32)
              a9 = numpy.concatenate(all_area9).ravel().astype(numpy.float32)
              a5 = numpy.concatenate(all_area5).ravel().astype(numpy.float32)
              current = time.time()
              # print("landmark ", current - start, "ms")

              # max thread per block is 1024, and max block per grid is 304, so be careful
              # calculating parallel using GPU
            #   conv33(drv.Out(r33r), drv.Out(r33i), drv.In(a33), drv.In(f33r), drv.In(f33i), block=(68,4,1), grid=(33,33))
            #   conv17(drv.Out(r17r), drv.Out(r17i), drv.In(a17), drv.In(f17r), drv.In(f17i), block=(68,4,1), grid=(17,17))
            #   conv9(drv.Out(r9r), drv.Out(r9i), drv.In(a9), drv.In(f9r), drv.In(f9i), block=(68,4,1), grid=(9,9))
              r33r = a33 * f33r
              r33i = a33 * f33i
              r17r = a17 * f17r
              r17i = a17 * f17i
              r9r = a9 * f9r
              r9i = a9 * f9i
              r5r = a5 * f5r
              r5i = a5 * f5i

              # accumulate value each filtersize^2 index
              splr33r = numpy.sum(numpy.split(r33r,272),axis = 1)
              splr33i = numpy.sum(numpy.split(r33i,272),axis = 1)
              splr17r = numpy.sum(numpy.split(r17r,272),axis = 1)
              splr17i = numpy.sum(numpy.split(r17i,272),axis = 1)
              splr9r = numpy.sum(numpy.split(r9r,272),axis = 1)
              splr9i = numpy.sum(numpy.split(r9i,272),axis = 1)
              splr5r = numpy.sum(numpy.split(r5r,272),axis = 1)
              splr5i = numpy.sum(numpy.split(r5i,272),axis = 1)

              # calculating magnitude of each pair filter
              mag33 = numpy.sqrt(splr33r**2 + splr33i**2)
              mag17 = numpy.sqrt(splr17r**2 + splr17i**2)
              mag9 = numpy.sqrt(splr9r**2 + splr9i**2)
              mag5 = numpy.sqrt(splr5r**2 + splr5i**2)

              # calculating phase of each pair filter
              phase33 = numpy.arctan(splr33i / splr33r)
              phase17 = numpy.arctan(splr17i / splr17r)
              phase9 = numpy.arctan(splr9i / splr9r)
              phase5 = numpy.arctan(splr5i / splr5r)

              # combine each same size magnitude and phase into 1
              featureall = numpy.concatenate((mag33, phase33, mag17, phase17, mag9, phase9, mag5, phase5))

              # Normalisasi ke 0 dan 1 sebelum masuk ke NN
              normalize = ((featureall - numpy.amin(featureall)) * 1) / ( numpy.amax(featureall) - numpy.amin(featureall))
              # convert to string before to dictionary
              normalize = normalize.astype('str')
              # convert dari numpy ke dictionary
              jsonall = dict(enumerate(normalize,1))
              # stream the data using kafka
              # stream(jsonall)

              frame = cv2.imencode(".jpg", frame[20:y-20,20:x-20])[1].tobytes()
              yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

              end = time.time()
              # print('progress : ',float(s)/len(subject90)*100, ' %')
              print('all time : ',end - start, ' ms')
              # waktu.append(end-start)
        # realtime
        # cv2.imshow("camera",cam)
        # cv2.waitKey(1)
      else:
        cap.release()
        cv2.destroyAllWindows()
        break

if __name__ == '__main__':
    app.run(debug=True)
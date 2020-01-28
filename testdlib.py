from itertools import chain
import cv2, numpy, dlib, os, time
import dlib.cuda as cuda
import pycuda.autoinit
import pycuda.driver as drv

cuda.set_device(0)
face_detector = dlib.get_frontal_face_detector()
cuda.set_device(0)
predictor = dlib.shape_predictor("Rachmad_ws/python/shape_predictor_68_face_landmarks.dat")
print("Hellocuda")
cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    _,frame = cap.read()
    frame = cv2.resize(frame,(480,360))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cuda.set_device(0)
    faces = face_detector(grey)
    for face in faces:
        all_area = []
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cuda.set_device(0)
        landmark = predictor(grey, face)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255),4)
        for i in range(0,68):
            x0 = landmark.part(i).x
            y0 = landmark.part(i).y
            cv2.circle(frame, (x0,y0), 1, (0,255,255),2)
            # get area 33 x 33 di setiap titik
            area33 = grey[y0-16:y0+17, x0-16:x0+17]
            # area17 = grey[y0-8:y0+9, x0-8:x0+9]
            # dst
            # shape all area (68,33,33)
            for dup in range(4):
                all_area.append(area33)

        # geser indentasi ke kanan agar dapat mendeteksi dan memproses multiple face secara simultan
        # all_area sudah siap untuk dikirim ke cuda, dengan dimensi 1 x 296208
        all_area = numpy.concatenate(all_area).ravel()
        # print(type(all_area))
        # cv2.imshow("frame",frame)
        # cv2.waitKey(1)
        current = time.time()
        print(1/(current - start), "fps")
        # if(key == 27):
        #     break


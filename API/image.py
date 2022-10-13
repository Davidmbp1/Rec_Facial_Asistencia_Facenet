from __future__ import print_function

import cv2
from scipy.spatial.distance import cosine
import mtcnn
from keras.models import load_model
from utils import *
import time
import os
import sys
import threading
import numpy as np
from datetime import datetime

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        # this lets the read() method block until there's a new frame
        self.cond = threading.Condition()

        # this allows us to stop the thread gracefully
        self.running = False

        # keeping the newest frame around
        self.frame = None

        # passing a sequence number allows read() to NOT block
        # if the currently available one is exactly the one you ask for
        self.latestnum = 0

        # this is just for demo purposes
        self.callback = None

        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet

        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1

                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)


def recognize(img,
              detector,
              encoder,
              encoding_dict,
              recognition_t=0.3,
              confidence_t=0.99,
              required_size=(160, 160), ):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist


        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img

if __name__ == '__main__':
    encoder_model = 'C:/Users/Usuario/PycharmProjects/Face-recognition-project-master/API/facenet_keras.h5'
    encodings_path = 'C:/Users/Usuario/PycharmProjects/Face-recognition-project-master/API/encodings/encodings.pkl'

    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)
    encoding_dict = load_pickle(encodings_path)

    # these windows belong to the main thread
    #cv2.namedWindow("frame")
    # on win32, imshow from another thread to this DOES work
    # cv2.namedWindow("realtime")

    # open some camera
    # cap = cv2.VideoCapture('rtsp://admin:Hik12345@192.168.0.100:554/Streaming/channels/101/') # Cámara del labo
    # cap = cv2.VideoCapture('rtsp://admin:Hik12345@192.168.20.116:554/Streaming/channels/101/') # Cámara de abajo
    # cap = cv2.VideoCapture(0) # Webcam
    cap = cv2.imread('C:/Users/Usuario/PycharmProjects/Face-recognition-project-master/API/assets/img/history/2022-9-27/Henrique De Aguiar/arrival.jpg')
    # cap.set(cv2.CAP_PROP_FPS, 30)

    IMG = recognize(cap, face_detector, face_encoder, encoding_dict)

    cv2.imwrite('prueba.png', IMG)
    # cv2.imshow("realtime_1", IMG)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("done!")

    # # wrap it
    # fresh = FreshestFrame(cap)
    #
    # # a way to watch the camera unthrottled
    # def callback(img):
    #     cv2.imshow("realtime", img)


    # main thread owns windows, does waitkey

    # fresh.callback = callback

    # main loop
    # get freshest frame, but never the same one twice (cnt increases)
    # see read() for details
    # cnt = 0
    # while True:
    #     # test that this really takes NO time
    #     # (if it does, the camera is actually slower than this loop and we have to wait!)
    #     t0 = time.perf_counter()
    #     cnt, img = fresh.read(seqnumber=cnt + 1)
    #     dt = time.perf_counter() - t0
    #     if dt > 0.010:  # 10 milliseconds
    #         print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))
    #
    #     # let's pretend we need some time to process this frame
    #     # print("processing {cnt}...".format(cnt=cnt), end=" ", flush=True)
    #     resize = cv2.resize(img, (800, 600))
    #
    #     while cnt == False:
    #         print("Can't receive frame. Retrying ...")
    #         cap.release()
    #         cap = cv2.VideoCapture('rtsp://admin:Hik12345@192.168.0.100:554/Streaming/channels/101/')
    #         # vc = cv2.VideoCapture(0)
    #         cnt, img = fresh.read(seqnumber=cnt + 1)
    #         dt = time.perf_counter() - t0
    #         if dt > 0.010:  # 10 milliseconds
    #             print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))
    #         resize = cv2.resize(img, (800, 600))
    #
    #     frame = recognize(resize, face_detector, face_encoder, encoding_dict)
    #
    #     cv2.imshow("realtime_1", frame)
    #     # this keeps both imshow windows updated during the wait (in particular the "realtime" one)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    #     print("done!")
    #
    # fresh.release()

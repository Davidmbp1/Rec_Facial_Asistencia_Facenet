from __future__ import print_function

import keras
import face_recognition
import numpy as np
import cv2
import queue
import threading
import time
import requests, os, re
import cv2
import mtcnn
import os
import sys
import numpy as np

from scipy.spatial.distance import cosine
from keras.models import load_model
from utils import *

import multiprocessing as mp


class Camera():

    def __init__(self, rtsp_url):
        # load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        # load process
        self.p = mp.Process(target=self.update, args=(child_conn, rtsp_url))
        # start process
        self.p.daemon = True
        self.p.start()

    def end(self):
        # send closure request to process

        self.parent_conn.send(2)

    def update(self, conn, rtsp_url):
        # load cam into seperate process

        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        print("Cam Loaded...")
        run = True

        while run:

            # grab frames from the buffer
            cap.grab()

            # recieve input data
            rec_dat = conn.recv()

            if rec_dat == 1:
                # if frame requested
                ret, frame = cap.read()
                conn.send(frame)

            elif rec_dat == 2:
                # if close requested
                cap.release()
                run = False

        print("Camera Connection Closed")
        conn.close()

    def get_frame(self, resize=0.65):
        ###used to grab frames from the cam connection process

        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

        # send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()

        # reset request
        self.parent_conn.send(0)

        # resize if needed
        if resize == None:
            return frame
        else:
            return self.rescale_frame(frame, resize)

    # def rescale_frame(self, frame, percent=0.65):
    #
    #     return cv2.resize(frame, None, fx=percent, fy=percent)
    # ########################## ---------------------------------------------------------- ########################################


############ Facenet utilizando multiprocesamiento para actualizar el buffer

# class FreshestFrame(threading.Thread):
#     def __init__(self, capture, name='FreshestFrame'):
#         self.capture = capture
#         assert self.capture.isOpened()
#
#         # this lets the read() method block until there's a new frame
#         self.cond = threading.Condition()
#
#         # this allows us to stop the thread gracefully
#         self.running = False
#
#         # keeping the newest frame around
#         self.frame = None
#
#         # passing a sequence number allows read() to NOT block
#         # if the currently available one is exactly the one you ask for
#         self.latestnum = 0
#
#         # this is just for demo purposes
#         self.callback = None
#
#         super().__init__(name=name)
#         self.start()
#
#     def start(self):
#         self.running = True
#         super().start()
#
#     def release(self, timeout=None):
#         self.running = False
#         self.join(timeout=timeout)
#         self.capture.release()
#
#     def run(self):
#         counter = 0
#         while self.running:
#             # block for fresh frame
#             (rv, img) = self.capture.read()
#             assert rv
#             counter += 1
#
#             # publish the frame
#             with self.cond:  # lock the condition for this operation
#                 self.frame = img if rv else None
#                 self.latestnum = counter
#                 self.cond.notify_all()
#
#             if self.callback:
#                 self.callback(img)
#
#     def read(self, wait=True, seqnumber=None, timeout=None):
#         # with no arguments (wait=True), it always blocks for a fresh frame
#         # with wait=False it returns the current frame immediately (polling)
#         # with a seqnumber, it blocks until that frame is available (or no wait at all)
#         # with timeout argument, may return an earlier frame;
#         #   may even be (0,None) if nothing received yet
#
#         with self.cond:
#             if wait:
#                 if seqnumber is None:
#                     seqnumber = self.latestnum + 1
#                 if seqnumber < 1:
#                     seqnumber = 1
#
#                 rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
#                 if not rv:
#                     return (self.latestnum, self.frame)
#
#             return (self.latestnum, self.frame)
# #############

########################## ---------------------------------------------------------- ########################################

############# Face recogntion y envio de datos a postgres usando dlib y la libreria Face_Recognition
# # Select the webcam of the computer
#
# # Encoding de rostros conocidos
# # video_capture = VideoCapture('https://stream-eu1-charlie.dropcam.com:443/nexus_aac/b85a6ec812c045cd921f4164e8e7ecc0/playlist.m3u8?public=GqJifk6U25')
# video_capture = VideoCapture(0)
#
# # video_capture.set(5,1)
#
# # * -------------------- USERS -------------------- *
# known_face_encodings = []
# known_face_names = []
# known_faces_filenames = []
#
# for (dirpath, dirnames, filenames) in os.walk('assets/img/users/'):
#     known_faces_filenames.extend(filenames)
#     break
#
# for filename in known_faces_filenames:
#     face = face_recognition.load_image_file('assets/img/users/' + filename)
#     known_face_names.append(re.sub("[0-9]", '', filename[:-4]))
#     known_face_encodings.append(face_recognition.face_encodings(face)[0])
#
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True
#
# # Fin del Encoding
#
# while True:
#     # for i in range(5):
#     #     video_capture.grab()
#     # Grab a single frame of video
#     frame = video_capture.read()
#
#     # # Resize frame of video to 1/4 size for faster face recognition processing
#     # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     # print(sys.exc_info())
#     # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#     # frame = small_frame[:, :, ::-1]
#
#     # Process every frame only one time
#     if process_this_frame:
#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)
#
#         # Initialize an array for the name of the detected users
#         face_names = []
#
#         # * ---------- Initialyse JSON to EXPORT --------- *
#         json_to_export = {}
#
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"
#
#             # # If a match was found in known_face_encodings, just use the first one.
#             # if True in matches:
#             #     first_match_index = matches.index(True)
#             #     name = known_face_names[first_match_index]
#
#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]
#
#                 # * ---------- SAVE data to send to the API -------- *
#                 json_to_export['name'] = name
#                 json_to_export['hour'] = f'{time.localtime().tm_hour}:{time.localtime().tm_min}'
#                 json_to_export[
#                     'date'] = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
#                 json_to_export['picture_array'] = frame.tolist()
#
#                 # * ---------- SEND data to API --------- *
#
#                 r = requests.post(url='http://127.0.0.1:3000/receive_data', json=json_to_export)
#                 print("Status: ", r.status_code)
#
#             face_names.append(name)
#
#     process_this_frame = not process_this_frame
#
#     # Display the results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#         # top *= 4
#         # right *= 4
#         # bottom *= 4
#         # left *= 4
#
#         # Draw a box around the face
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#         # Draw a label with a name below the face
#         # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#
#     # Display the resulting image
#     cv2.imshow('Video', frame)
#
#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release handle to the webcam
# video_capture.release()
# cv2.destroyAllWindows()
#
# ###############

########################## ---------------------------------------------------------- ########################################

############## Inicio del reconocimiento usando Facenet y guardamos la Data para luego ser enviada a la API

def recognize(img,
              detector,
              encoder,
              encoding_dict,
              recognition_t=0.5,
              confidence_t=0.99,
              required_size=(160, 160), ):
    face_names = []
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

                json_to_export = {}

                # * ---------- SAVE data to send to the API -------- *
                json_to_export['name'] = name
                json_to_export['hour'] = f'{time.localtime().tm_hour}:{time.localtime().tm_min}'
                json_to_export['date'] = f'{time.localtime().tm_year}-{time.localtime().tm_mon}-{time.localtime().tm_mday}'
                time.sleep(2000)
                json_to_export['picture_array'] = img.tolist()

                # * ---------- SEND data to API --------- *

                r = requests.post(url='http://127.0.0.1:3000/receive_data', json=json_to_export)
                print("Status: ", r.status_code)

            face_names.append(name)

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 200), 2)
    return img

################# Fin del reconocimiento



if __name__ == '__main__':
    encoder_model = 'C:/Face-recognition-project-master/API/facenet_keras.h5'
    encodings_path = 'C:/Face-recognition-project-master/API/encodings/encodings.pkl'

    face_detector = mtcnn.MTCNN()
    face_encoder = load_model(encoder_model)
    encoding_dict = load_pickle(encodings_path)



    cam = Camera(0)

    print(f"Camera is alive?: {cam.p.is_alive()}")

    while (1):
        img = cam.get_frame(0.65)

        frame = recognize(img, face_detector, face_encoder, encoding_dict)

        cv2.imshow("Rec_f", frame)

        key = cv2.waitKey(1)

        if key == 13:  # 13 is the Enter Key
            break

    cv2.destroyAllWindows()

    cam.end()




    # # these windows belong to the main thread
    # #cv2.namedWindow("frame")
    # # on win32, imshow from another thread to this DOES work
    # cv2.namedWindow("realtime")
    #
    # # open some camera
    # # cap = cv2.VideoCapture('rtsp://admin:Hik12345@192.168.20.116:554/Streaming/channels/101/')
    # cap = VideoCapture(0)
    # # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # # cap.set(cv2.CAP_PROP_FPS, 20)
    #
    # # # wrap it
    # # fresh = FreshestFrame(cap)
    # #
    # # # a way to watch the camera unthrottled
    # # def callback(img):
    # #     cv2.imshow("realtime", img)
    #
    #
    # # main thread owns windows, does waitkey
    #
    # # fresh.callback = callback
    #
    # # main loop
    # # get freshest frame, but never the same one twice (cnt increases)
    # # see read() for details
    # cnt = 0
    # while True:
    #     # # test that this really takes NO time
    #     # # (if it does, the camera is actually slower than this loop and we have to wait!)
    #     # t0 = time.perf_counter()
    #     img = cap.read()
    #     # dt = time.perf_counter() - t0
    #     # if dt > 0.010:  # 10 milliseconds
    #     #     print("NOTICE: read() took {dt:.3f} secs".format(dt=dt))
    #     #
    #     # # let's pretend we need some time to process this frame
    #     # print("processing {cnt}...".format(cnt=cnt), end=" ", flush=True)
    #
    #     frame = recognize(img, face_detector, face_encoder, encoding_dict)
    #
    #     cv2.imshow("realtime_1", frame)
    #     # this keeps both imshow windows updated during the wait (in particular the "realtime" one)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    #     print("done!")
    #
    # fresh.release()

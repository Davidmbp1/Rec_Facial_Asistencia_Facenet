# * ---------- IMPORTACION --------- *
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import psycopg2
import cv2
import numpy as np
import re
import pyttsx3
import time


# Get the relativ path to this file (we will use it later)
FILE_PATH = os.path.dirname(os.path.realpath(__file__))

# * ---------- CREANDO LA App --------- *
app = Flask(__name__)
CORS(app, support_credentials=True)
#
# # * ---------- CREDENCIALES DE LA BASE DE DATOS --------- *
# DATABASE_USER = os.environ['username']
# DATABASE_PASSWORD = os.environ['username']
# DATABASE_HOST = os.environ['username']
# DATABASE_PORT = os.environ['username']
# DATABASE_NAME = os.environ['username']

def DATABASE_CONNECTION():
    return psycopg2.connect(user="postgres",
                            password="123456789",
                            host="localhost",
                            port="5432",
                            database="Attendance_LAB_SC")


# * --------------------  ROUTES ------------------- *
# * ---------- OBTENER LA DATA DEL RECONOCIMIENTO FACIAL ---------- *
@app.route('/receive_data', methods=['POST'])
def get_receive_data():
    if request.method == 'POST':
        json_data = request.get_json()

        # VERIFICAR SI EL USUARIO YA SE ENCUENTRA EN LA BASE DE DATOS
        try:
            # CONEXION CON LA BASE DE DATOS
            connection = DATABASE_CONNECTION()
            cursor = connection.cursor()

            # QUERY PARA VERIFICAR SI EL USUARIO HA SIDO VISTO POR LA CAMARA EL DIA ACTUAL
            user_saw_today_sql_query = \
                f"SELECT * FROM users WHERE date = '{json_data['date']}' AND name = '{json_data['name']}'"

            cursor.execute(user_saw_today_sql_query)
            result = cursor.fetchall()
            connection.commit()

            # SI EL USUARIO SE ENCUENTRA EN LA BASE DE DATOS EL DIA ACTUAL

            count = 0

            if result:
                print('user IN')


                if json_data['status'] == "0":

                # AUDIO  #######################################################
                    name = json_data['name']
                    data = "Nos vemos {}".format(name)


                    s = pyttsx3.init()


                    s.say(data)

                    rate = s.getProperty('rate')
                    s.setProperty('rate', 150)

                    s.runAndWait()

                    time.sleep(1)

                    # finsh AUDIO###################################################

                    json_data['status'] = "1"

                if json_data['status']=="1":

                    #name = json_data['name']
                    #data = "Bienvenido devuelta, {}".format(name)

                    # language = 'es'
                    #
                    # myobj = gTTS(text=data, lang=language, slow=False)
                    #
                    # myobj.save("welcome.mp3")
                    #
                    # os.system("welcome.mp3")

                    s = pyttsx3.init()
                    name = json_data['name']
                    data = "Bienvenido devuelta, {}".format(name)
                    s.say(data)

                    rate = s.getProperty('rate')
                    s.setProperty('rate', 150)

                    s.runAndWait()
                    time.sleep(1)

                    json_data['status'] = "0"

                image_path = f"{FILE_PATH}/assets/img/{json_data['date']}/{json_data['name']}/departure.jpg"

                # Save image
                os.makedirs(f"{FILE_PATH}/assets/img/{json_data['date']}/{json_data['name']}", exist_ok=True)
                cv2.imwrite(image_path, np.array(json_data['picture_array']))
                json_data['picture_path'] = image_path


                # Update user in the DB
                update_user_querry = f"UPDATE users SET departure_time = '{json_data['hour']}', departure_picture = '{json_data['picture_path']}', status = {json_data['status']} WHERE name = '{json_data['name']}' AND date = '{json_data['date']}'"
                cursor.execute(update_user_querry)
                connection.commit()
                #time.sleep(1)


            else:
                print("user OUT")
                # Save image

                # AUDIO  #######################################################

                s = pyttsx3.init()
                name = json_data['name']
                data = "Bienvenido {}, a Smart city".format(name)
                s.say(data)

                rate = s.getProperty('rate')
                s.setProperty('rate', 150)

                s.runAndWait()

                # finsh AUDIO###################################################


                image_path = f"{FILE_PATH}/assets/img/history/{json_data['date']}/{json_data['name']}/arrival.jpg"
                os.makedirs(f"{FILE_PATH}/assets/img/history/{json_data['date']}/{json_data['name']}", exist_ok=True)
                cv2.imwrite(image_path, np.array(json_data['picture_array']))
                json_data['picture_path'] = image_path
                json_data['status'] = '0'

                # Create a new row for the user today:
                insert_user_querry = f"INSERT INTO users (name, date, arrival_time, arrival_picture, status) VALUES ('{json_data['name']}', '{json_data['date']}', '{json_data['hour']}', '{json_data['picture_path']}', {json_data['status']})"
                cursor.execute(insert_user_querry)
                connection.commit()
                #time.sleep(1)


        except (Exception, psycopg2.DatabaseError) as error:
            print("ERROR DB: ", error)
        finally:
            connection.commit()

            # closing database connection.
            if connection:
                cursor.close()
                connection.close()
                print("PostgreSQL connection is closed")

        # Return user's data to the front
        return jsonify(json_data)


# * ---------- Get all the data of an employee ---------- *
@app.route('/get_employee/<string:name>', methods=['GET'])
def get_employee(name):
    answer_to_send = {}
    # Check if the user is already in the DB
    try:
        # Connect to DB
        connection = DATABASE_CONNECTION()
        cursor = connection.cursor()
        # Query the DB to get all the data of a user:
        user_information_sql_query = f"SELECT * FROM users WHERE name = '{name}'"

        cursor.execute(user_information_sql_query)
        result = cursor.fetchall()
        connection.commit()

        # if the user exist in the db:
        if result:
            print('RESULT: ', result)
            # Structure the data and put the dates in string for the front
            for k, v in enumerate(result):
                answer_to_send[k] = {}
                for ko, vo in enumerate(result[k]):
                    answer_to_send[k][ko] = str(vo)
            print('answer_to_send: ', answer_to_send)
        else:
            answer_to_send = {'error': 'User not found...'}

    except (Exception, psycopg2.DatabaseError) as error:
        print("ERROR DB: ", error)
    finally:
        # closing database connection:
        if (connection):
            cursor.close()
            connection.close()

    # Return the user's data to the front
    return jsonify(answer_to_send)


# * --------- Get the 5 last users seen by the camera --------- *
@app.route('/get_5_last_entries', methods=['GET'])
def get_5_last_entries():
    answer_to_send = {}
    # Check if the user is already in the DB
    try:
        # Connect to DB
        connection = DATABASE_CONNECTION()

        cursor = connection.cursor()
        # Query the DB to get all the data of a user:
        lasts_entries_sql_query = f"SELECT * FROM users ORDER BY id DESC LIMIT 5;"

        cursor.execute(lasts_entries_sql_query)
        result = cursor.fetchall()
        connection.commit()

        # if DB is not empty:
        if result:
            # Structure the data and put the dates in string for the front
            for k, v in enumerate(result):
                answer_to_send[k] = {}
                for ko, vo in enumerate(result[k]):
                    answer_to_send[k][ko] = str(vo)
        else:
            answer_to_send = {'error': 'error detect'}

    except (Exception, psycopg2.DatabaseError) as error:
        print("ERROR DB: ", error)
    finally:
        # closing database connection:
        if (connection):
            cursor.close()
            connection.close()

    # Return the user's data to the front
    return jsonify(answer_to_send)


# * ---------- Add new employee ---------- *
@app.route('/add_employee', methods=['POST'])
@cross_origin(supports_credentials=True)
def add_employee():
    try:
        # Get the picture from the request
        image_file = request.files['image']
        print(request.form['nameOfEmployee'])

        # Store it in the folder of the know faces:
        file_path = os.path.join(f"assets/img/users/{request.form['nameOfEmployee']}.jpg")
        image_file.save(file_path)
        answer = 'new employee succesfully added'
    except:
        answer = 'Error while adding new employee. Please try later...'
    return jsonify(answer)


# * ---------- Get employee list ---------- *
@app.route('/get_employee_list', methods=['GET'])
def get_employee_list():
    employee_list = {}

    # Walk in the user folder to get the user list
    walk_count = 0
    for file_name in os.listdir(f"{FILE_PATH}/assets/img/users/"):
        # Capture the employee's name with the file's name
        name = re.findall("(.*)\.jpg", file_name)
        if name:
            employee_list[walk_count] = name[0]
        walk_count += 1

    return jsonify(employee_list)


# * ---------- Delete employee ---------- *
@app.route('/delete_employee/<string:name>', methods=['GET'])
def delete_employee(name):
    try:
        # Remove the picture of the employee from the user's folder:
        print('name: ', name)
        file_path = os.path.join(f'assets/img/users/{name}.jpg')
        os.remove(file_path)
        answer = 'Employee succesfully removed'
    except:
        answer = 'Error while deleting new employee. Please try later'

    return jsonify(answer)


# * -------------------- RUN SERVER -------------------- *
if __name__ == '__main__':
    # * --- DEBUG MODE: --- *
    app.run(host='127.0.0.1', port=3000, debug=True)
    #  * --- DOCKER PRODUCTION MODE: --- *
    # app.run(host='0.0.0.0', port=os.environ['PORT']) -> DOCKER
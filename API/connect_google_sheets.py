import psycopg2
import pandas as pd
import gspread

from oauth2client import client # Added
from oauth2client import tools # Added
from oauth2client.file import Storage # Added
from gspread_dataframe import set_with_dataframe

try:
    connection=psycopg2.connect(
        host='localhost',
        user='postgres',
        port=5432,
        password='123456789',
        database='Attendance_LAB_SC'
    )
    print('conexion exitosa')
except Exception as ex:
    print(ex)

sql="SELECT * FROM users"
df=pd.read_sql(sql,connection)

credencial={"installed":{"client_id":"761482497104-6df0to44debsbjarjni9pohi6fp2vbtf.apps.googleusercontent.com",
                         "project_id":"smartcity-face-recognition",
                         "auth_uri":"https://accounts.google.com/o/oauth2/auth",
                         "token_uri":"https://oauth2.googleapis.com/token",
                         "auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs",
                         "client_secret":"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","redirect_uris":["http://localhost"]}}
gc,auth_user=gspread.oauth_from_dict(credencial)
sh=gc.open_by_key('1IaElq7f9fNC9lneaCIbTtRM76p8yfKhZXsoMGOHSn68')
worksheet=sh.get_worksheet(0)
set_with_dataframe(worksheet,df)

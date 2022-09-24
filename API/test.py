import psycopg2
try:
    connection=psycopg2.connect(
        host='localhost',
        user='postgres',
        password='8182',
        database='attendance'
    )
    print('conexion exitosa')
    cursor=connection.cursor()
    cursor.execute("SELECT version()")
    row=cursor.fetchone()
    print(row)
    cursor.execute("SELECT * FROM users")
    usuarios=cursor.fetchall()
    print(usuarios)
    cursor.close()
    connection.close()
except Exception as ex:
    print(ex)
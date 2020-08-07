import sqlite3
from sqlite3 import Error

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        print('table created')
    except Error as e:
        print(e)

def init_table():
    sql_create_employee_table = """ CREATE TABLE IF NOT EXISTS employees (
                                        id integer PRIMARY KEY,
                                        registered_number text,
                                        name text
                                    ); """

    sql_create_image_table = """ CREATE TABLE IF NOT EXISTS images (
                                        id integer PRIMARY KEY,
                                        user_id integer,
                                        sesi text,
                                        pose text,
                                        path text
                                    ); """

    if conn is not None:
        # create employees table
        create_table(conn, sql_create_employee_table)
        # create images table
        create_table(conn, sql_create_image_table)
    else:
        print("Error! cannot create the database connection.")

def create_employee(conn , employee):
    """
    Create a new employee into the employee table
    :param conn:
    :param employee:
    :return:
    """

    sql = ''' INSERT INTO employees(id,registered_number,name)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, employee)
    conn.commit()
    return cur.lastrowid

def create_image(conn , image):
    """
    Create a new image into the image table
    :param conn:
    :param image:
    :return:
    """

    sql = ''' INSERT INTO images(id, user_id, sesi, pose, path)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, image)
    conn.commit()
    return cur.lastrowid

def select_all_employee(conn):
    """
    Query all rows in the employee table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM employees")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def select_all_image(conn):
    """
    Query all rows in the image table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM images")

    rows = cur.fetchall()

    for row in rows:
        print(row)

def delete_employee_and_its_images(conn, id):
    sql = 'DELETE FROM employees WHERE id=?'
    sql_image = 'DELETE FROM images WHERE user_id=?'
    cur = conn.cursor()
    cur.execute(sql, (id,))
    conn.commit()
    cur.execute(sql_image, (id,))
    conn.commit()

def select_employee_and_images(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM employees LEFT JOIN images ON employees.id=images.user_id")

    rows = cur.fetchall()

    for row in rows:
        print(row)

if __name__ == '__main__':
    conn = create_connection(r"/home/er2c-jetson-nano/data_sqlite/data_karyawan.db")
    init_table()
    with conn:
        # create a new project
        employee = (1, 1, 'ikbar')
        image = (2, 1, 'A','+30','')
        # create_employee(conn, employee)
        # create_image(conn, image)
        # select_employee_and_images(conn)
        # select_all_image(conn)
        delete_employee_and_its_images(conn,1)

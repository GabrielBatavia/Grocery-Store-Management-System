import mysql.connector

__cnx = None

def get_sql_connection():
    global __cnx
    __cnx = mysql.connector.connect(host="localhost", user="root", password="", database="gs", port=3306)
    return __cnx


if __name__ == '__main__':
    try:
        connection = get_sql_connection()
        if connection:
            print("Connected to MySQL successfully!")
            # Example usage:
            # print(get_all_products(connection))
    except mysql.connector.Error as err:
        print(f"Error in main execution: {err}")
    finally:
        if __cnx:
            __cnx.close()

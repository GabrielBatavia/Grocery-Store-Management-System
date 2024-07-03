import mysql.connector
from sql_connection import get_sql_connection

def get_all_product(connection):
    cursor = connection.cursor()

    query = """
    SELECT product.product_id, product.product_name, product.uom_id, product.price, uom.uom_name 
    FROM product 
    INNER JOIN uom ON product.uom_id = uom.uom_id
    """

    cursor.execute(query)

    response = []

    for (product_id, product_name, uom_id, price, uom_name) in cursor:
        response.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "uom_id": uom_id,
                "price": price,
                "uom_name": uom_name
            }
        )

    return response

def insert_new_product(connection, product):
    cursor = connection.cursor()
    query = (
        """
        INSERT INTO product
        (product_name, uom_id,price)
        VALUES (%s, %s, %s)
        """
    )

    data = (product['product_name'], product['uom_id'], product['price'])
    cursor.execute(query, data)
    connection.commit()
    
    return cursor.lastrowid

def delete_product(connection, product_id):
    cursor = connection.cursor()
    query = ("DELETE FROM product where product_id=" + str(product_id))
    cursor.execute(query)
    connection.commit()
    
if __name__ == '__main__':
    try:
        connection = get_sql_connection()
        if connection:
            print("Connected to MySQL successfully!")
            # Example usage
            products = get_all_product(connection)
            print("Products retrieved:", products)
            
            new_product_id = delete_product(connection, 0)
    except mysql.connector.Error as err:
        print(f"Error in main execution: {err}")
    finally:
        if connection and connection.is_connected():
            connection.close()
            print("Connection closed.")

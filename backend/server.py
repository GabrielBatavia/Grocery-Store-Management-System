# .\env\Scripts\activate


from flask import Flask, render_template, url_for, request, redirect, jsonify
from datetime import datetime
from sql_connection import get_sql_connection
import products_dao

app = Flask(__name__)

connection = get_sql_connection()

@app.route('/getProducts', methods=['GET'])
def get_products():
    products = products_dao.get_all_product(connection)
    
    response = jsonify(products)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    app.run(port=5000, debug=True)
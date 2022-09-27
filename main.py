
from unittest import result
from flask_cors import CORS
from flask import Flask
from flask import request, jsonify
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import json

from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
CORS(app)
#app.config['SECRET_KEY'] = 'supersecret'
app.config['UPLOAD_FOLDER'] = './static'


df = pd.DataFrame()
chart_array = []


@app.route('/')
def welcome():
    return 'welcome'


@app.route('/file_upload', methods=['GET', 'POST'])
def fileUpload():

    file = request.files.get('file')
    global df
    # df=pd.read_csv(file,parse_dates=True)
    filename = secure_filename(file.filename)

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    msg = "success"
    return jsonify({"response": msg})


@app.route('/train', methods=['GET'])
def train_data():
    df = pd.read_csv('./static/jagane.csv')
    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    df['forecast'] = results.predict(start=50, end=103, dynamic=True)
    predicted_sales1 = df['forecast'][50:103].to_numpy()
    l0 = predicted_sales1.tolist()
    predicted_sales = json.dumps(l0)

    actual_sales1 = df['Sales'][50:103].to_numpy()
    l1 = actual_sales1.tolist()
    actual_sales = json.dumps(l1)

    dates1 = df['Date'][50:103].to_numpy()
    l2 = dates1.tolist()
    dates = json.dumps(l2)

    return {'dates': dates, 'actual_sales': actual_sales, 'predicted_sales': predicted_sales}


if __name__ == '__main__':
    app.run(debug=True)

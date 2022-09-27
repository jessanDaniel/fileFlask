
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
    df = pd.read_csv("./static/jagane.csv", parse_dates=True)
    # df.columns=["Month","Sales"]
    # df.plot()
    df.dropna()
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(12)
    model = ARIMA(df['Sales'], order=(1, 1, 1))
    model_fit = model.fit()
    df['forecast'] = model_fit.predict(start=50, end=103, dynamic=True)
    # df[['Sales','forecast']].plot(figsize=(12,8))
    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    df['forecast'] = results.predict(start=50, end=103, dynamic=True)

    # df = df.set_index(['Date'])
    # df[['Sales','forecast']].plot(figsize=(12,8))
    # chart_array.append(df['forecast'][50:103])

    json_response1 = df['forecast'][50:103].to_numpy()
    s = json_response1.tolist()
    json_response = json.dumps(s)

    # json_response1 = df['forecast'][50:103].to_json()

    dates_df = df['Date'][50:103].to_numpy()
    l = dates_df.tolist()
    dates = json.dumps(l)

# if its a encoded JSON ---convert that to string---> var str = JSON.stringify(categories);

# Replace slash everywhere in the string----------->str =str.replace(/\//g,"");

# You can convert back to JSON object again using-->var output =JSON.parse(str);

    # print(dates)

    return {'sales': json_response, 'dates': dates}


if __name__ == '__main__':
    app.run(debug=True)

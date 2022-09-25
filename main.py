
from flask_cors import CORS
from flask import Flask
from flask import request, jsonify
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
CORS(app)
#app.config['SECRET_KEY'] = 'supersecret'
app.config['UPLOAD_FOLDER'] = './static'


df=pd.DataFrame()
chart_array=[]

@app.route('/')
def welcome():
    return 'welcome'


@app.route('/file_upload', methods=['GET', 'POST'])
def fileUpload():

    file = request.files.get('file')
    global df
    #df=pd.read_csv(file,parse_dates=True)
    filename = secure_filename(file.filename)
    
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    msg = "success"
    return jsonify({"response": msg})

@app.route('/train', methods=['GET'])
def train_data():
    df=pd.read_csv("./static/jagane.csv",parse_dates=True)
    #df.columns=["Month","Sales"]
    #df.plot()
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(12)
    model=ARIMA(df['Sales'],order=(1,1,1))
    model_fit=model.fit()
    df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
    #df[['Sales','forecast']].plot(figsize=(12,8))
    model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
    results=model.fit()
    df['forecast']=results.predict(start=90,end=103,dynamic=True)
    #df[['Sales','forecast']].plot(figsize=(12,8))
    chart_array.append(df['forecast'][50:103])
    resp={"chart":chart_array}
    
    print(chart_array)
    return resp
    
    

if __name__ == '__main__':
    app.run(debug=True)


from sklearn.metrics import r2_score, mean_absolute_percentage_error

from flask_cors import CORS
from flask import Flask
from flask import request
import pandas as pd
import re
from flask_pymongo import pymongo
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
from datetime import date
import json
import numpy as np

from werkzeug.utils import secure_filename
import os
# * global variables
df = pd.DataFrame()
filename = ''
rmse = ''
accuracy = ''
MAPE = ''
# ******
app = Flask(__name__)
CORS(app)
#app.config['SECRET_KEY'] = 'supersecret'
app.config['UPLOAD_FOLDER'] = './static'
ALLOWED_EXTENSIONS = {'csv'}
# * database purpose***********
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


con_string = "mongodb+srv://jessandaniel:jessandaniel@cluster0.bvq4qwg.mongodb.net/?retryWrites=true&w=majority"


client = pymongo.MongoClient(con_string)

db = client.get_database('sales_prediction')

user_collection = pymongo.collection.Collection(
    db, 'Sales_Prediction_users')  # (<database_name>,"<collection_name>")
print("MongoDB connected Successfully")
# ***************


@app.route('/')
def welcome():
    return 'welcome'


@app.route('/create-user', methods=['POST'])
def register_user():
    msg = ''
    try:
        req_body = request.get_json(force=True)
        var = req_body['email']
        if re.fullmatch(regex, req_body['email']):
            if (not user_collection.find_one({"email": var})):
                # Hashing password using message digest5 algorithm
                password = req_body['password']
                hashed_password = hashlib.sha256(
                    password.encode('utf-8')).hexdigest()
                # username and passwords are inserted to mongoDB using insert_one function
                user_collection.insert_one(
                    {"email": var, "password": hashed_password})
                msg = 'SignUp Successful'
            else:
                msg = 'User Already Exists'
        else:
            msg = 'Mail is not an email'

    except Exception as e:
        print(e)
        msg = 'User Already Exists'
    return {'resp': msg}


# ***************
# * login route


@app.route('/login', methods=['POST'])
def signin():
    msg = ''
    try:
        data = request.get_json(force=True)
        print(data)
        var = data['email']

        # Hashing password using message digest5 algorithm
        password = data['password']
        hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
        # username and password are comapred with mongoDB using find_one function
        out = user_collection.find_one(
            {"email": var, "password": hashed_password})
        u1 = out.get('email')
        p1 = out.get('password')

        msg = 'Login Successful'
    except Exception as e:
        print(e)
        msg = 'Please check your credentials'
    return {'resp': msg}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/file_upload', methods=['GET', 'POST'])
def fileUpload():
    msg = ''
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:

            msg = 'File not attached'

        file = request.files.get('file')
        # If the user does not select a file, the browser submits an
        # !empty file without a filename.
        if file.filename == '':
            msg = 'Please select a file'

        if file and allowed_file(file.filename):
            global filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            msg = 'file upload success'
        else:
            msg = 'select a csv file'
    return {'response': msg}

    # df=pd.read_csv(file,parse_dates=True)
    # filename = file.filename

    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # msg = "success"
    # return {"response": msg}


@app.route('/user_input', methods=['POST'])
def train_data():

    data = request.get_json(force=True)
    fromm = data['from']
    to = data['to']

    global filename
    path = 'static/'+filename
    df = pd.read_csv(path, parse_dates=True, index_col='Date')
    df = df.dropna()

    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    df['forecast'] = results.predict(start=50, end=103, dynamic=True)
    predicted_sales1 = df['forecast'][50:103].to_numpy()
    predicted_sales2 = predicted_sales1.tolist()
    predicted_sales = json.dumps(predicted_sales2)

    actual_sales1 = df['Sales'][50:103].to_numpy()
    actual_sales2 = actual_sales1.tolist()
    actual_sales = json.dumps(actual_sales2)

    dates1 = list(df.index[50:103])
    dates2 = [str(date)[:-9] for date in dates1]
    dates = json.dumps(dates2)

    # to from difference >0,

    future_sales = []
    final = str(df.index[-1])[:-9].split('-')
    final_date = date(int(final[0]), int(final[1]), int(final[2]))
    date_dict = {}
    # * calculating difference in dates or future prediction
    date_dict['from'] = (fromm.split('-'))
    date_dict['to'] = (to.split('-'))

    from_date = date(int(date_dict['from'][0]), int(
        date_dict['from'][1]), int(date_dict['from'][2]))
    to_date = date(int(date_dict['to'][0]), int(
        date_dict['to'][1]), int(date_dict['to'][2]))
    difference = (to_date-from_date).days
    extra_days = (to_date-final_date).days
    start = extra_days-difference
    end = extra_days

    # *setting extra dates*********************
    future_dates = [df.index[-1]+DateOffset(days=x)
                    for x in range(0, extra_days+1)]
    future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_date_df])
    future_df['forecast'] = results.predict(
        start=start+105, end=105+end+1, dynamic=False)
    # ***************************

    future_sales1 = future_df['forecast'][105+start:].to_numpy()
    future_sales2 = future_sales1.tolist()
    future_sales = json.dumps(future_sales2)

    future_user_date1 = list(future_df.index[-difference-1:])
    future_user_date2 = [str(date)[:-9] for date in future_user_date1]
    future_user_date = json.dumps(future_user_date2)

    # * rmse calculation
    mse = np.square(np.subtract(df['Sales'], df['forecast'])).mean()
    rmse_unparsed = np.sqrt(mse)
    global rmse
    rmse = json.dumps(rmse_unparsed)
    #***********************#

    # *accuracy calculation
    accuracy_unparsed = r2_score(df.Sales[70:103], df.forecast[70:103])
    accuracy_unparsed = str(accuracy_unparsed*100)[:5]+'%'
    global accuracy
    accuracy = json.dumps(accuracy_unparsed)
    # **************

    # * MAPE
    mape_unparsed = mean_absolute_percentage_error(
        df.Sales[70:103], df.forecast[70:103])
    global MAPE
    MAPE = json.dumps(mape_unparsed)

    resp = {'actual': actual_sales,
            'predicted': predicted_sales, 'dates': dates, 'future_user_date': future_user_date, 'future_sales': future_sales}

    return resp


@app.route('/custom_value', methods=['POST'])
def custom_prediction():
    data = request.get_json(force=True)
    custom_unparsed_date = data['custom_date']

    global filename
    path = './static/'+filename
    df = pd.read_csv(r'./static/'+filename, parse_dates=True, index_col='Date')
    df = df.dropna()
    # * training
    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    final = str(df.index[-1])[:-9].split('-')
    final_date = date(int(final[0]), int(final[1]), int(final[2]))
    # * parsing into a date
    date_dict = {}
    date_dict['date'] = custom_unparsed_date.split('-')
    # * finding length between last date to custom date
    custom_parsed_date = date(int(date_dict['date'][0]), int(
        date_dict['date'][1]), int(date_dict['date'][2]))
    extra_days = (custom_parsed_date-final_date).days

    # * adding the extra date
    future_dates = [df.index[-1]+DateOffset(days=x)
                    for x in range(0, extra_days+1)]
    future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_date_df])
    future_df['forecast'] = results.predict(
        start=105, end=105+extra_days+1, dynamic=False)
    predicted_non_json_value = future_df['forecast'][-1]
    predicted_value = json.dumps(predicted_non_json_value)

    return {'custom_value': predicted_value[:7]}


@app.route('/additional_days', methods=['POST'])
def additional_days():
    data = request.get_json(force=True)
    days = int(data['days'])
    global filename
    path = './static/'+filename
    df = pd.read_csv(path, parse_dates=True, index_col='Date')
    df = df.dropna()

    model = sm.tsa.statespace.SARIMAX(
        df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    # to from difference >0,

    future_sales = []

    extra_days = days

    end = extra_days

    # setting extra dates********
    future_dates = [df.index[-1]+DateOffset(days=x)
                    for x in range(0, extra_days+1)]
    future_date_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_date_df])
    future_df['forecast'] = results.predict(
        start=105, end=105+end+1, dynamic=False)
    # *********

    future_sales1 = future_df['forecast'][105:].to_numpy()
    future_sales2 = future_sales1.tolist()
    future_sales = json.dumps(future_sales2)

    future_user_date1 = list(future_df.index[-days:])
    future_user_date2 = [str(date)[:-9] for date in future_user_date1]
    future_user_date = json.dumps(future_user_date2)

    resp = {'future_user_date': future_user_date, 'future_sales': future_sales}

    return resp


@app.route('/accuracy_and_error', methods=['GET'])
def stats():
    global rmse
    global accuracy
    global MAPE
    if rmse == '' or accuracy == '' or MAPE == '':
        msg = 'failure'
    elif rmse != '' and accuracy != '' and MAPE != '':
        msg = 'success'
    return {'status': msg, 'rmse': rmse[:4], 'accuracy': accuracy[1:5], 'MAPE': MAPE[:5]}


if __name__ == '__main__':
    app.run(debug=True)

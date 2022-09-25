
from flask import Flask
from flask import request

from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
#app.config['SECRET_KEY'] = 'supersecret'
app.config['UPLOAD_FOLDER'] = './static'


@app.route('/')
def welcome():
    return 'welcome'


@app.route('/file_upload', methods=['GET', 'POST'])
def fileUpload():
    file = request.files.get('file')

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return 'file successfully uploaded'


if __name__ == '__main__':
    app.run(debug=True)

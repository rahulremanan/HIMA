import config
import logging
from flask import current_app, Flask, render_template, request, flash, \
send_file, send_from_directory
import urllib
import base64
import json
from werkzeug import secure_filename
import os
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import random
import string
import gc

app = Flask(__name__)

UPLOAD_FOLDER = '/home/rahul/Jomiraki/upload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
SERVER_URL = 'http://192.168.2.2:8080'
APP_URL = 'http://192.168.2.2:5000'

app.config.from_object(config)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SERVER_URL'] = SERVER_URL
app.config['APP_URL'] = APP_URL

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fetch_predictions(img_stream=None):
    img_stream=img_stream
    ENCODING = 'utf-8'
    server_url = app.config['SERVER_URL'] + '/async/'
    base64_string = base64.b64encode(img_stream)
    r = requests.post(server_url, files={'file': base64_string})
    print(r.text)
    return r

def fetch_predictions_filename(filename=None):
    filename=filename
    server_url = app.config['SERVER_URL'] + '/async/upload/'
    file = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
    files = {'file': open(file, 'rb')}
    r = requests.post(server_url, files=files)
    del files
    gc.collect()
    print(r.text)
    return r

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        img = request.files.get('file')
        N = 64
        filename = (''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.png')
        if img and allowed_file(img.filename):
            print ('Received file: ... ' + secure_filename(img.filename))
            print (filename)
            img_stream = img.read()
            try:
                img_data= Image.open(BytesIO(img_stream))
            except OSError as e:
                flash ("Unable to read image data ...")
                return render_template('form.html')                
            img_data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            predictions = fetch_predictions_filename(filename=filename)
            print (predictions.json()["Location"])
            del img_data
            del img_stream
            del img
            gc.collect()
            return render_template('view.html', \
                               image_url=app.config['APP_URL']+ '/uploads/' \
                               + filename,\
                               predictions=predictions.json()["Location"])
        else:
            flash ("No image file selected ...")
            return render_template('form.html')
    else:
        flash("Image file not supported ...")
        return render_template('form.html')

@app.route('/uploads/<filename>')
def fetch_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.errorhandler(500)
def server_error(e):
    logging.error('An error occurred during a request.')
    return 'An internal error occurred.', 500

if __name__=="__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(host='0.0.0.0', port=5000, debug=True)
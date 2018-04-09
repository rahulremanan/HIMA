#!/usr/bin/python3.6
import config
import logging
from flask import   current_app,        \
                    Flask,              \
                    render_template,    \
                    request,            \
                    flash,              \
                    send_file,          \
                    send_from_directory 
import urllib
import base64
import json
from werkzeug import secure_filename
import os
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request,     \ 
                           urlopen
import random
import string
import gc

app = Flask(__name__)

UPLOAD_FOLDER = '/home/rahul/Jomiraki/upload/'                                  # Change this to the correct local directory ...
SERVER_URL = 'http://192.168.2.4:8000'                                          # Change this to the API server url ...
APP_URL = 'http://192.168.2.4:80'                                               # Change this to the app server url ...

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'gif'] 

app.config.from_object(config)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SERVER_URL'] = SERVER_URL
app.config['APP_URL'] = APP_URL
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

logging.basicConfig(level=logging.INFO)

def is_valid_dir(dir_name):
    if not os.path.isdir(dir_name):
        print ("The folder: %s does not exist ..." % dir_name)
        print ("Select an output folder location ...")
    else:
        print ("Found a folder: &s ..." % dir_name)
        return dir_name

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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
    try:
        file_data = {'file': open(file, 'rb')}
    except:
        print ("Failed opening file ...")
        return jsonify(status_code='400', msg='Failed opening file ...')
    try:
        r = requests.post(server_url, files=file_data)
    except:
        print ("Failed posting request ...")
        return jsonify(status_code='400', msg='Failed posting a request ...')
    del file_data
    gc.collect()
    print(r.text)
    return r

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        img = request.files.get('file')
        N = 64
        filename = (''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N)) + '.png')
        if img != None and allowed_file(img.filename):
            print ('File received: ... ' + secure_filename(img.filename))
            print (filename)
            img_stream = img.read()
            img.close()
            del img
            gc.collect()
            try:
                img_data= Image.open(BytesIO(img_stream))
                del img_stream
                gc.collect()
            except OSError as e:
                print ("Failed reading image data ...")
                return render_template('form.html')
            try:
                img_data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            except:
                print ("Failed saving image data ...")
                return render_template('form.html')
            img_data.close()
            del img_data
            gc.collect()
            try:
                predictions = fetch_predictions_filename(filename=filename)
                print (predictions.json()["Location"])
            except:
                print ("Failed creating predictions ...")
                return render_template('form.html')
            return render_template('view.html', \
                               image_url=app.config['APP_URL']+ '/uploads/' \
                               + filename,\
                               prediction_loc=predictions.json()["Location"],\
                               predictions=predictions.json())
        else:
            flash ("No image file selected ...")
            return render_template('form.html')
    else:
        flash ("File format not supported ...")
        return render_template('form.html')

@app.route('/uploads/<filename>')
def fetch_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(500)
def server_error(e):
    logging.error('An error occurred during a request.')
    return render_template('form.html')

@app.errorhandler(500)
def server_error(e):
    logging.error('An error occurred during a request.')
    return 'An internal error occurred.', 500

if __name__=="__main__":
    is_valid_dir(app.config['UPLOAD_FOLDER'])
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', port=80, debug=False)
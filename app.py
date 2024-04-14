from flask import Flask, render_template, request, send_file
import cv2
from utils import Arnold, generateChirikovMap, padding, encrypt
import numpy as np
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/encrypt', methods=['POST','GET'])
def upload_file():
    file = request.files['file']
    password = request.form.get('text')
    img = Image.open(file)
    img = np.array(img)
    encrypted_img = encrypt(password, img)
    base64_encrypted = base64.encode(encrypted_img)
    return 'hello'

@app.route('/decrypt', methods=['POST', 'GET'])
def show_img():
    return 'why?'

if __name__ == '__main__':
    app.run(debug=True)

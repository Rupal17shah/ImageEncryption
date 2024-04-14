from flask import Flask, render_template, request, send_file
import cv2
from utils import Arnold, generateChirikovMap, padding, encrypt, decrypt
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import matplotlib.pyplot as plt
import os 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/encrypt', methods=['POST','GET'])
def upload_file():
    file = request.files['file']
    password = request.form.get('text')
    print(password)
    img = Image.open(file)
    img = np.array(img).astype('uint8')
    encrypted_img = encrypt(password, img)
    np.save(os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_img.npy'), encrypted_img)
    print("----------------- printing image-------------------")
    print(encrypted_img)
    pil_image = Image.fromarray(encrypted_img.astype('uint8'))
    pil_image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_img.jpg'))
    return render_template('encrypt.html', img_path=os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_img.jpg'))

@app.route('/decrypt', methods=['POST', 'GET'])
def show_img():
    password = request.form.get('text')
    print(password)
    img = np.load(os.path.join(app.config['UPLOAD_FOLDER'], 'encrypted_img.npy'))
    decrypted_img = decrypt(password, img)
    message = "Decryption Successful! Type in correct password to decrypt the original image."
    pil_image = Image.fromarray(decrypted_img)
    pil_image.save(os.path.join(app.config['UPLOAD_FOLDER'], 'decrypted_img.jpg'))
    return render_template('encrypt.html', img_path=os.path.join(app.config['UPLOAD_FOLDER'], 'decrypted_img.jpg'), message=message)

if __name__ == '__main__':
    app.run(debug=True)

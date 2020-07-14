import os
import time

from flask import Flask
from flask import request
from flask import render_template
from model_utils import ModelConfig, predict_class

app = Flask(__name__, static_folder='resources/user_images')
UPLOAD_FOLDER = 'resources/user_images/'


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image_name = f'{int(time.time())}.jpg'
        image_file = request.files['image']
        image_loc = os.path.join(UPLOAD_FOLDER, image_name)
        image_file.save(image_loc)

        probs, names = predict_class(image_loc, ModelConfig)
        
        return render_template('index.html', res=zip(probs, names), image_name=image_name)
    return render_template('index.html', res=None, image_name=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
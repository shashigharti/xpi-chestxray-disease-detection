#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, jsonify
import os
import fastai
from fastai import *
from fastai.vision import *
from settingsc import *
import torchvision
import torchvision.transforms as transforms


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['THRESHOLD'] = 10


@app.route('/')
def index():
    """
        Render front page
    """
    return render_template('index.html')


@app.route('/upload-image', methods=['POST'])
def results():
    """
        Predict the uploaded image
    """
    if request.method == 'POST':
        image = request.files['file']
        image.save(os.path.join(app.config['UPLOAD_FOLDER'],
                                image.filename))
        img = open_image(os.path.join(app.config['UPLOAD_FOLDER'],
                                      image.filename))

        (pred_class, pred_idx, outputs) = learn.predict(img)
        app.logger.info(pred_class, pred_idx, outputs)
        app.logger.info('Image %s classified as %s' % (image.filename,
                                                       pred_class))

        predicted_classes_with_probabilities = [[labels[idx],
                                                 str(round(probability.item() * 100, 2))] for (idx,
                                                                                               probability) in
                                                enumerate(outputs) if probability * 100
                                                > app.config['THRESHOLD']]
        app.logger.info(predicted_classes_with_probabilities)

        response = {'name': image.filename,
                    'class': predicted_classes_with_probabilities}
        return render_template('results.html', result=response)

# Run inference on cpu
path = 'static/chestxray'
fastai.device = torch.device('cpu')
data = ImageDataBunch.single_from_classes(path, labels, size=224)
learn = cnn_learner(data, models.resnet50)

# Check if model exists and load
path_to_model = os.path.join(dir, 'models', 'stage-1-rn50-50000.pth')
print(path_to_model)
if not os.path.exists(path_to_model):
    print ('\nmodel not found')
else:
    learn = learn.load('stage-1-rn50-50000')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


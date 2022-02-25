import os

from flask import Flask, render_template, request, redirect, url_for

from inference import get_prediction
from commons import get_model

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/<class_type>/', methods=['GET', 'POST'])
def chooseModel(class_type):
    if request.method == 'POST':
        model_name = request.form.get('model')
        return redirect(url_for("testing", model=model_name))
    return render_template('choice_model.html', class_name=class_type)


@app.route('/test/', methods=['GET', 'POST'])
def testing():
    model_ins = get_model()
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        conf, iou = None, None
        if request.form.get("conf"):
            conf = request.form.get("conf")
        if request.form.get("iou"):
            iou = request.form.get("iou")
        if not file:
            return
        img_bytes = file.read()
        class_name, probability = get_prediction(image_bytes=img_bytes, model_ins=model_ins, conf=conf, iou=iou)
        return render_template('result.html', class_name=class_name, proba=probability*100)
    return render_template('testing.html')


if __name__ == '__main__':
    app.run(debug=True)

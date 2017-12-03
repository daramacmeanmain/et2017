import os
import flask as fl 
import tensorflow as tf
import numpy as np
from flask import render_template, request, g, redirect, url_for
from flask import send_from_directory
from PIL import Image
import PIL.ImageOps

from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'imageUploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = fl.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def root():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))

@app.route('/imageUploads/<filename>')
def uploaded_file(filename):
    im = Image.open(filename)
    img = im.resize((28, 28))
    grey = img.convert("L")
    greyInvert = PIL.ImageOps.invert(grey)
    arr = np.array(greyInvert)

    SummariesDirectory = "C:/Users/Dara/Desktop/emergingTech/project/etProject2017/mnist_checkpoints/"
    ModelName = "mnist-2900"
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(SummariesDirectory + ModelName + ".meta")
    new_saver.restore(sess, SummariesDirectory + ModelName)

    y_conv = tf.get_collection("y_conv")[0]
    x = tf.get_collection("x")[0]
    y_ = tf.get_collection("y_")[0]
    keep_prob =  tf.get_collection("keep_prob")[0]

    InputImage = arr.reshape(1,784)

    logit = sess.run(y_conv,feed_dict={ x: InputImage, keep_prob: 1.0})
    prediction = sess.run(tf.argmax(logit,1))

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run()
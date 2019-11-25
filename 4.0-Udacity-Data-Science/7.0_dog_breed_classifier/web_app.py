import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
import tensorflow as tf
import dog_breed

app = Flask(__name__)

def setup_app(app):
    # All your initialization code
    app.config['UPLOAD_FOLDER'] = 'images/'
    app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'png', 'jpg', 'jpeg',
                                        'gif'}
    app.config['dog_detector'] = dog_breed.DogBreedDetector()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    app.config['dog_detector'].detect_breed(app.config['UPLOAD_FOLDER'] + filename)
    print("Show image:", filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    print(tf.__version__)
    setup_app(app)
    app.run(host='0.0.0.0', port=3001, debug=True)

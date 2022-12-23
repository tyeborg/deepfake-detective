from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from deepfake_detective import DeepfakeDetective
import os
import dlib

app = Flask(__name__ , template_folder='static/templates')
@app.route('/')
def main():
  return render_template ("video_deepfake.html")

# Upload File 
UPLOAD_FOLDER = 'static'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/predict_video', methods=['POST'])
def upload_video():
	f = request.files['file']
	filename = secure_filename(f.filename)
	f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	filepath = "static/" + filename

    deepfake_detective = DeepfakeDetective(filepath)
    video_fps = deepfake_detective.fps
    print(f'Video FPS: {video_fps}')

if __name__ =='__main__': 
  app.run(port=3000, debug=True)
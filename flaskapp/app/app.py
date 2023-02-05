# Import the appropriate libraries for future use.
import os
from wtforms import FileField
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

# Import the DeepfakeDetective class.
from deepfake_detective import DeepfakeDetective, ImageInput, VideoInput
from config import Config, ProductionConfig, DevelopmentConfig, TestingConfig

# Establish the valid file extension types for users to upload.
ALLOWED_EXTENSIONS = set(['mp4', 'mov', 'jpeg', 'jpg', 'png'])

# Construct a method to determine if an uploaded file has a valid extension.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_files_folder():
    files_dir = 'static/files'

    # Create files folder in case it does not already exist.
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    # Clear the files folder.
    for f in os.listdir(files_dir):
        os.remove(os.path.join(files_dir, f))

def create_app():
    # Create an app instance.
    app = Flask(__name__ , template_folder='templates')

    # Attach our configurations to the app object.
    # Dynamically change type of configurations depending on the FLASK ENV.
    if app.config["ENV"] == "production":
        app.config.from_object("config.ProductionConfig")
    elif app.config["ENV"] == "testing":
        app.config.from_object("config.TestingConfig")
    else:
        app.config.from_object("config.DevelopmentConfig")

    # Create a Home route.
    @app.route('/', methods=['GET', 'POST'])
    @app.route('/uploadImage', methods=['GET', 'POST'])
    def uploadImage():
        if request.method == 'POST':
            # Grab the file from the input field.
            f = request.files['file']

            # Handle 'files' Folder in 'static' folder.
            handle_files_folder()

            # Determine if file exists and contains a valid extension.
            if f and allowed_file(f.filename):
                # Convert filename to a secured filename (removes slashes from filename).
                filename = secure_filename(f.filename)
                # Save the file object in the static/files location.
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                # Initialize a filepath for the stored input videofile.
                filepath = app.config['UPLOAD_FOLDER'] + "/" + filename

                # Execute DeepfakeDetective with the given videofile.
                deepfake_detective = ImageInput(filepath)

                real, fake = deepfake_detective.predict()
                return(f'REAL: {real}% | FAKE: {fake}%')

        # Return the content of 'video_deepfake.html' onto the home webpage.
        return render_template ("index.html")

    return app
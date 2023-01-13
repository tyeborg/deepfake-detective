# pip install flask
# pip install flask_wtf
# pip install wtforms
# pip install python-magic-bin==0.4.14
# pip install -U pylint --user

# Import the appropriate libraries for future use.
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField
import os
import dlib
import magic

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

# Create a method that will determine if the file type of the input file.
def get_filetype(filename):
    # Identify the file type by checking input file header.
    file_type = magic.from_file(filename, mime=True)
    # Split the string from the index where a slash is located.
    file_type = file_type.split("/")
    # Return the first item within the list (file type).
    return file_type[0]

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
    @app.route('/upload', methods=['GET', 'POST'])
    def main():
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
                deepfake_detective = DeepfakeDetective(filepath)

                file_type = deepfake_detective.get_file_type()

                if file_type == "image":
                    print("Dealing with IMAGE input")
                    deepfake_detective = ImageInput(filepath)
                elif file_type == "video":
                    print("Dealing with VIDEO input")
                    deepfake_detective = VideoInput(filepath)
                else:
                    deepfake_detective = None

                if deepfake_detective != None:
                    is_valid = deepfake_detective.validate()

                    if is_valid == "success":
                        real, deepfake = deepfake_detective.predict()
                        real, deepfake = round(round(real, 2) * 100), round(round(deepfake, 2) * 100)
                        print(f'[*] REAL: {real}%')
                        print(f'[*] DEEPFAKE: {deepfake}%')
                        return(f'REAL: {real}% | DEEPFAKE: {deepfake}%')
                    else:
                        error_msg = deepfake_detective.validate()[1]
                        return(f'{error_msg}')
                        #for msg in error_msg:
                            #print(f'[*] {msg}')

        # Return the content of 'video_deepfake.html' onto the home webpage.
        return render_template ("index.html")

    return app

if __name__ =='__main__':
    app = create_app()
    app.run(port=3000, debug=True)
# Import the appropriate libraries for future use.
import os
import shutil
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

# Create a function that creates and maintains a folder to store all 
# frames to be evaluated by the deepfake detector.
def handle_files_folder():
    folder_path = 'app/static/files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Create empty subfolders within the folder.
    subfolder_paths = [
        os.path.join(folder_path, 'img'),
        os.path.join(folder_path, 'video')
    ]
    for subfolder_path in subfolder_paths:
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            
    # Clear out all files in the folder and subfolders
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            os.unlink(os.path.join(root, file))
        
# Create a function that reduces the length of the filename.
def reduce_filename(filename):
    # Alter the filename depending if it is too long or not.
    if(len(filename) >= 6):
        split_name = filename.split('.')
        filename = split_name[0][0:6] + "... ." + split_name[1]
        
    return filename

# Create a function that validates the file for an appropriate extension.
def validate_file(app, request, filename):
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
        
    return(filename)

# Create a function that determines the classification of image or video.
def receive_results(content):
    # Utilize the predict funtion to obtain results.
    real, fake = content.predict()

    # Initialize a variable to stringify the results to send to HTML.
    if real > fake:
        results = 'REAL'
    else:
        results = 'FAKE'
        
    return(results)
    
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
    def home():
        return render_template('home2.html')
    
    # Create the Deepfake Detective Tool (image) route.
    @app.route('/deepfake-detective', methods=['GET', 'POST'])
    @app.route('/deepfake-detective/image-upload', methods=['GET', 'POST'])
    def uploadImage():
        # Initialize variables.
        proof, results, filename, val_response = None, None, None, None
        
        # Execute only if a POST request was received.
        if request.method == 'POST': 
            # Obtain a filepath if the file has been validated.    
            filename = validate_file(app, request, filename)
            
            # Determine if the file was validated.
            if filename != None:
                # Initialize a filepath for the stored input image file.
                filepath = app.config['UPLOAD_FOLDER'] + "/" + filename
                
                # Execute DeepfakeDetective with the given videofile.
                img = ImageInput(filepath)
                
                # Determine if the image meets the evaluation criteria.
                if img.num_faces == 0:
                    val_response = f'Error: Unable to detect a face from {filename}.'
                elif img.num_faces > 1:
                    val_response = f'Error: Multiple faces detected from {filename}.'
                else:
                    # Reduce the filename to send to HTML.
                    filename = reduce_filename(filename)
                    # State a success response if successful
                    val_response = f'Success: {filename} has met the evaluation criteria.'
                    
                    # Obtain the proof image returned from the deepfake detective.
                    proof = 'app/static/files/img/frame.jpg'
                    
                    # Obtain the results of either 'REAL' or 'FAKE'.
                    results = receive_results(img)

        # Return the content to the webpage.
        return render_template('official.html', proof=proof, results=results, filename=filename, val_response=val_response)
    
    # Create the Deepfake Detective Tool (video) route.
    @app.route('/deepfake-detective', methods=['GET', 'POST'])
    @app.route('/deepfake-detective/video-upload', methods=['GET', 'POST'])
    def uploadVideo():
        # Initialize variables.
        proof, results, filename, val_response = None, None, None, None
        
        # Execute only if a POST request was received.
        if request.method == 'POST': 
            # Obtain a filepath if the file has been validated.    
            filename = validate_file(app, request, filename)
            
            # Determine if the file was validated.
            if filename != None:
                # Initialize a filepath for the stored input image file.
                filepath = app.config['UPLOAD_FOLDER'] + "/" + filename
                
                # Reduce the filename to send to HTML.
                filename = reduce_filename(filename)

                # Execute DeepfakeDetective with the given video file.
                video = VideoInput(filepath)
                
                # Determine if the video meets the evaluation criteria.
                if video.num_faces == 0:
                    val_response = f'Error: Unable to detect a face from {filename}.'
                elif video.num_faces > 1:
                    val_response = f'Error: Multiple faces detected from {filename}.'
                elif video.video_duration < 120:
                    val_response = f'Error: {filename} happens to be too long to analyze.'
                else:
                    # State a success response if successful
                    val_response = f'Success: {filename} has met the evaluation criteria.'
                
                    # Obtain the proof image returned from the deepfake detective.
                    proof = 'app/static/files/video/movie.mp4'
                    # Reduce the filename to send to HTML.
                    filename = reduce_filename(filename)
                    
                    # Obtain classification results.
                    results = receive_results(video)

        # Return the content to the webpage.
        return render_template('official.html', proof=proof, results=results, filename=filename, val_response=val_response)
    
    return app
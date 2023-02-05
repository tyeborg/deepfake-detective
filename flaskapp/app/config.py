class Config():
    DEBUG = False
    TESTING = False
    # Set our Secret Key.
    SECRET_KEY = "secretkey"
    # Configure upload folder where all the files will be uploaded into.
    UPLOAD_FOLDER = 'static/files'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    # While in dev env we want to debug our app using Flask interactive debugger.
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
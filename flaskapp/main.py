import sys
# Navigate to the app/app.py that contains the create_app method.
sys.path.insert(1, './app')
import app

if __name__ =='__main__':
    webapp = app.create_app()
    webapp.run(host="0.0.0.0", port=int("3000"), debug=True)
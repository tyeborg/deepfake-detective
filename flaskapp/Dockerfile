# Use the Python3.7 container image.
FROM python:3.7-stretch

# Set the working directory to /app.
WORKDIR /app

# Copy the current directory contents into the container at /app.
ADD . /app

# Install the dependencies.
RUN pip install setuptools 
RUN pip install -r requirements.txt

# Expose our application port (3000).
EXPOSE 3000

# Run the command to start application.
CMD python ./main.py
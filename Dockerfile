# Use an official lightweight Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Manually install tf-keras to ensure compatibility with RetinaFace
RUN pip install tf-keras

# Install additional libraries necessary for OpenCV to function
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Expose the port that the Flask app runs on
EXPOSE 5000

# Set the environment variable for Flask (optional)
ENV FLASK_APP=app.py

# Command to run your Flask application
CMD ["python", "app.py"]

# Use the Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install eventlet
RUN pip install eventlet

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app 
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on 
EXPOSE 5000

# Define environment variable 
ENV FLASK_APP=app.py


# Run the Flask app with SocketIO and Eventlet
CMD ["python", "app.py"]


# Run the Flask application
#CMD ["flask", "run", "--host=0.0.0.0"]







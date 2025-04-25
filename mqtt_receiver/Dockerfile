# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from this directory to WORKDIR
COPY . .

# Run the script
CMD ["python3", "-u", "mqtt_receiver.py"]

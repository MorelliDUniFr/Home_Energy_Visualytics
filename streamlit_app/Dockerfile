# Use an official Python image from Docker Hub
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Add the streamlit_app directory to the Python path
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

# Expose the port that the Streamlit app will run on
EXPOSE 8501

# Set the environment variable for Streamlit
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0"]


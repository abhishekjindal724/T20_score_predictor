<<<<<<< HEAD
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install wget and unzip for downloading and unzipping datasets
RUN apt-get update && apt-get install -y wget unzip

# Download multiple zip files from GitHub
RUN wget https://github.com/abhishekjindal724/T20_score_predictor/blob/main/t20s.zip -O t20s.zip
RUN wget https://github.com/abhishekjindal724/T20_score_predictor/blob/main/pipe.zip -O pipe.zip
RUN wget https://github.com/abhishekjindal724/T20_score_predictor/blob/main/dataset_level1.zip -O dataset_level1.zip

# Unzip all the downloaded datasets
RUN unzip t20s.zip -d /app/datasets/t20s
RUN unzip pipe.zip -d /app/datasets/pipe
RUN unzip dataset_level1.zip -d /app/datasets/dataset_level1

# Remove the zip files after unzipping
RUN rm dataset1.zip dataset2.zip dataset3.zip

# Expose the port that Streamlit uses (8501 by default)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
=======
# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit uses (8501 by default)
EXPOSE 8501

# Run the Streamlit app when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
>>>>>>> c44a98faf6e9c99b83c56c9caf77df8732f2bf28

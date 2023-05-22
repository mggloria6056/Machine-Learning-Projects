# Use a Python 3 base image
FROM python:3

# Copy the exported model file into the Docker image
COPY modeldensenet.h5 /
COPY modelogonet.h5 /
COPY modelxception.h5 /
COPY requirements.txt /
COPY 3.jpeg /

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install any additional dependencies that the model requires
RUN pip install -r requirements.txt

# Set the entry point command to a Python script that loads the model and performs inference
COPY Logo.py /
CMD [ "python", "Logo.py" ]
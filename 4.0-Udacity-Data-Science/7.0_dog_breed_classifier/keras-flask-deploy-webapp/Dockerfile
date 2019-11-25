FROM python:3.6-slim-stretch
RUN apt-get update && apt-get install -y procps vim libglib2.0 libsm6 libxext6 libxrender-dev

COPY . /root
WORKDIR /root

RUN pip install Werkzeug Flask numpy Keras gevent pillow h5py tensorflow tqdm opencv-python


EXPOSE 5000 3000
CMD [ "python" , "dog_breed_app.py"]


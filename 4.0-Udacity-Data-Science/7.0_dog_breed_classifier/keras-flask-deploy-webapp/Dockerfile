FROM python:2.7.16-slim-stretch
RUN apt-get update && apt-get install -y procps vim libglib2.0 libsm6 libxext6 libxrender-dev

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install Werkzeug Flask numpy Keras gevent pillow h5py tensorflow tqdm opencv-python


EXPOSE 5000 3000
CMD [ "python" , "dog_breed_app.py"]


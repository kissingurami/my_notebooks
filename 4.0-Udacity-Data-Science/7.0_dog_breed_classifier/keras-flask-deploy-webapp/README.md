# Deploy Keras Model with Flask as Web App in 10 Minutes

[![](https://img.shields.io/badge/python-2.7%2C%203.5%2B-green.svg)]()

> Dog breed classifier inference flash+keras web server

------------------

:point_down:Screenshot:

<p align="center">
  <img src="https://raw.githubusercontent.com/kissingurami/my_notebooks/master/4.0-Udacity-Data-Science/7.0_dog_breed_classifier/dog_breed_image_classifier_train/Screenshot.png" width="600px" alt="">
</p>

------------------

## Docker Installation

### Build and run an image for keras-application pretrained model 
```shell
$ cd keras-flask-deploy-webapp
$ docker build -t dog_breed_classifier 
$ docker run --name dog_app --detach -p 3000:3000 dog_breed_classifier
```

### Build and run an image from your model into the containeri.
After build an image as above, and 

### Pull an built-image from Docker hub
For your convenience, can just pull the image instead of building it. 
```shell
$ docker pull docker.io/kissingurami/dog_breed_classifier:1.0
$ docker run --name dog_app --detach -p 3000:3000 kissingurami/dog_breed_classifier:1.0
$ docker logs -f dog_app
```
Open http://localhost:3000 after waiting for a minute to install in the container.


## Local Installation

### Clone the repo
```shell
$ git clone https://github.com/mtobeiyf/keras-flask-deploy-webapp.git
```

### Install requirements

```shell
$ pip install -r requirements.txt
```

Make sure you have the following installed:
- tensorflow
- keras
- flask
- pillow
- h5py
- gevent

### Run with Python

Python 2.7 or 3.5+ are supported and tested.

```shell
$ python dog_breed_app.py
```

### Play

Open http://localhost:3000 and have fun. :smiley:

------------------

#!/bin/bash

name=keras-flask-dog-breed-webapp
docker stop $name
docker rm $name
docker run --name $name --detach \
        -p 3000:3000 \
        -p 5000:5000 \
        keras_flask_app 

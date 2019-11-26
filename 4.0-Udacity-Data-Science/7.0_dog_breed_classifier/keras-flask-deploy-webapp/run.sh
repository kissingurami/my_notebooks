#!/bin/bash

name=dog
docker stop $name
docker rm $name
docker run --name $name --detach \
        -p 3000:3000 \
        -p 5000:5000 \
        -p 2222:2222 \
        dog_breed_classifier

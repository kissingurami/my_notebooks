#!/bin/bash

name=train
docker stop $name
docker rm $name
docker run --name $name --detach \
        -p 2222:2222 \
        train

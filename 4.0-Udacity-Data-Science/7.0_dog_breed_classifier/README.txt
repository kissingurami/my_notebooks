# Dog Breed Classifier with Keras
This is a repo for the Dog Breed Classifier Project  in Udacity Nanodegree
Consist of 2 parts:
  * model training
  * inference web app
It is implemented by using numpy, Keras, tensorflow, flask

**Udacity's original repo is [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)**

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI  Nanodegree! In this project, you will learn how to build a pipeline that  can be used within a web or mobile app to process real-world,  user-supplied images.  Given an image of a dog, your algorithm will  identify an estimate of the canine’s breed.  If supplied an image of a  human, the code will identify the resembling dog breed.

[![Sample Output](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png)](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/project-dog-classification/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification  and localization, you will make important design decisions about the  user experience for your app.  Our goal is that by completing this lab,  you understand the challenges involved in piecing together a series of  models designed to perform various tasks in a data processing pipeline.   Each model has its strengths and weaknesses, and engineering a  real-world application often involves solving many problems without a  perfect answer.  Your imperfect solution will nonetheless create a fun  user experience!


## Part 1: Model Traning

### 1.1 Source code dir structure
*Generate by running: $ tree --charset=ascii*

```
|-- dog_breed_image_classifier_train
|   |-- README.md
|   |-- data
|   |-- dog_app.html
|   |-- dog_app.ipynb
|   |-- images
|   |   |-- Brittany_02625.jpg
|   |   |-- sample_human_2.png
|   |-- requirements.txt
|   `-- saved_models
|       |-- weights.best.VGG16.hdf5
|       |-- weights.best.Xception.hd5
|       `-- weights.best.from_scratch.hdf5
```
**dog_app.ipynb** is the main code to train the model. Models are saved under **saved_models**. You can check **dog_app.html** for the result.

### 1.2 Import Links
* Kaggle Compitition
```https://www.kaggle.com/c/dog-breed-identification```
* Download the dog dataset
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip```
* Download the human dataset
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip```

### 1.3 Nural Network Structures and evaluation

#### 1.3.1 Self made network 1

```
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 55, 55, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 54, 54, 64)        8256      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 27, 27, 64)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 133)               8645      
=================================================================
Total params: 19,189
Trainable params: 19,189
Non-trainable params: 0
_________________________________________________________________
```
-----

​	After trained for **20 epochs**, training accuracy is **5.76%**, testing accuracy is only **5.86%**.

#### 1.3.2. Self made network 2
```
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 224, 224, 4)  304         input_3[0][0]                    
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 112, 112, 4)  0           conv2d_4[0][0]                   
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 56, 56, 4)    148         max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 56, 56, 8)    40          conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 56, 56, 8)    40          conv2d_5[0][0]                   
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 56, 56, 4)    0           conv2d_5[0][0]                   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 56, 56, 8)    584         conv2d_6[0][0]                   
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 56, 56, 8)    1608        conv2d_8[0][0]                   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 56, 56, 8)    40          max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 56, 56, 24)   0           conv2d_7[0][0]                   
                                                                 conv2d_9[0][0]                   
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_7 (MaxPooling2D)  (None, 28, 28, 24)   0           concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 28, 28, 4)    100         max_pooling2d_7[0][0]            
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 3136)         0           conv2d_11[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 133)          417221      flatten_1[0][0]                  
==================================================================================================
Total params: 420,085
Trainable params: 420,085
Non-trainable params: 0
```

​	After trained for **20 epochs**, training accuracy is **99.45%**, testing accuracy is only **5.38%**. The model is **overfitting**.

#### 1.3.3 Transfer Learnings
The model uses the the pre-trained model as a fixed feature extractor, where the last convolutional output is fed as input to our model. We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.
```python
def transfer_learning(network):
    if network =='ResNet-50':
        bottleneck_features_network = np.load('data/bottleneck_features/DogResnet50Data.npz')
    elif network == 'InceptionV3':
        bottleneck_features_network = np.load('data/bottleneck_features/DogInceptionV3Data.npz')
    elif network =='Xception':
        bottleneck_features_network = np.load('data/bottleneck_features/DogXceptionData.npz')
    elif network =='VGG-16':
        bottleneck_features_network = np.load('data/bottleneck_features/DogVGG16Data.npz')
    elif network =='VGG-19':
        bottleneck_features_network = np.load('data/bottleneck_features/DogVGG19Data.npz')

    train_network = bottleneck_features_network['train']
    valid_network = bottleneck_features_network['valid']
    test_network = bottleneck_features_network['test']

    ### Define your architecture.
    Network_model = Sequential()
    Network_model.add(GlobalAveragePooling2D(input_shape=train_network.shape[1:]))
    Network_model.add(Dense(133, activation='softmax'))
    
    ### Compile the model.
    Network_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    ### Train the model.
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)
    ## Remember --> train_targets, valid_targets, test_targets contains the one-hot encoded correct values.
    Network_model.fit(train_network, train_targets,
              validation_data=(valid_network, valid_targets),
              epochs=10, batch_size=1, callbacks=[checkpointer], verbose=1)

    ### Load the model weights with the best validation loss.
    Network_model.load_weights(model_filepath)
    print("The best weights for the {} model have been loaded".format(network))

    ### Calculate classification accuracy on the test dataset.
    predictions = [np.argmax(Network_model.predict(np.expand_dims(feat, axis=0))) for feat in test_network]
    test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
```
**Our model:**
```
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
global_average_pooling2d_4 ( (None, 512)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 133)               68229     
=================================================================
Total params: 68,229
Trainable params: 68,229
Non-trainable params: 0
```
**Transfer learning bottleneck features**
    * RestNet50: 77M
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz```
    * VGG16: 811M
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz```
    * VGG19: 811M
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz```
    * Xception: 3.1G
```wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz```
    * InceptionV3: 1.6G
```https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz```

**Results:**
* transfer_learning('VGG-19')
​	After trained for **20 epochs**, training accuracy is **83.68%**, testing accuracy is up to **67.10%**.
* transfer_learning('VGG-16')
​	After trained for **20 epochs**, training accuracy is **99.24%**, testing accuracy is up to **70.93%**.
* transfer_learning('ResNet-50')
​	After trained for **20 epochs**, training accuracy is **88.28%**, testing accuracy is up to **77.63%**.
* transfer_learning('Inception')
​	After trained for **20 epochs**, training accuracy is **87.60%**, testing accuracy is up to **78.59%**.
* transfer_learning('Xception')
​	After trained for **20 epochs**, training accuracy is **87.60%**, testing accuracy is up to **84.21%**.

**Conclusion:**
ResNet50 achive pretty good result with less features file size as 77M. 
Xception achieve the best accuracy, but feature size is 3.1G.

## Part 2: Dog breed classifier inference web server(flash+keras)
### 2.1 Source code dir structure
*Generate by running: $ tree --charset=ascii*
```
`-- keras-flask-deploy-webapp
    |-- Dockerfile
    |-- README.md
    |-- app.py
    |-- build.sh
    |-- data
    |   `-- dogImages
    |-- dog_breed.py
    |-- dog_breed_app.py
    |-- models
    |   |-- weights.best.VGG16.hdf5
    |   |-- weights.best.VGG19.hdf5
    |   |-- weights.best.Xception.hd5
    |   `-- weights.best.from_scratch.hdf5
    |-- requirements.txt
    |-- run.sh
    |-- static
    |   |-- css
    |   |   `-- main.css
    |   `-- js
    |       `-- main.js
    |-- templates
    |   |-- base.html
    |   `-- index.html
    `-- uploads
        `-- README.md
```



:point_down:Screenshot:

<p align="center">
  <img src="https://raw.githubusercontent.com/kissingurami/my_notebooks/master/4.0-Udacity-Data-Science/7.0_dog_breed_classifier/dog_breed_image_classifier_train/Screenshot.png" width="600px" alt="">
</p>

### 2.2 Docker Installation

#### Build and run an image for keras model 
```shell
$ cd keras-flask-deploy-webapp
$ docker build -t dog_breed_classifier 
$ docker run --name dog_app --detach -p 3000:3000 dog_breed_classifier
```

#### Pull an built-image from Docker hub
For your convenience, can just pull the image instead of building it. 
```shell
$ docker pull docker.io/kissingurami/dog_breed_classifier:1.0
$ docker run --name dog_app --detach -p 3000:3000 kissingurami/dog_breed_classifier:1.0
$ docker logs -f dog_app
```
Open http://localhost:3000 after waiting for a minute to install in the container.




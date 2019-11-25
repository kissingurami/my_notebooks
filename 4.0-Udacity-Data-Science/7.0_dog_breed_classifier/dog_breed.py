
import numpy as np
from glob import glob
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]
print('There are %d total dog categories.' % len(dog_names))


import random
random.seed(8675309)

# utilities
from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# extract pre-trained face detector
import cv2
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# extract pre-trained isDog detector
# define ResNet50 model
from keras.applications.resnet50 import ResNet50
ResNet50_model = ResNet50(weights='imagenet')

from keras.applications.resnet50 import preprocess_input

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


#load pretrain dog breed detector network
network = 'Xception'
input_shape = (7, 7, 2048)
print('input_shape', input_shape)
# define low half network
Network_model = Sequential()
Network_model.add(GlobalAveragePooling2D(input_shape=input_shape))
Network_model.add(Dense(133, activation='softmax'))
Network_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_filepath = 'saved_models/weights.best.' + network + '.hd5'
Network_model.load_weights(model_filepath)
print("The best weights for the {} model have been loaded".format(network))

from extract_bottleneck_features import *
def network_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Network_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def detect_breed(image_path):
    print(image_path)
    if face_detector(image_path):
        print("Hello, human!")
    elif dog_detector(image_path):
        print("Hello, dog!")
    else:
        print("Error: Neither a human face or a dog was detected.\n")
        return
    print("You look like a ...")
    print(network_predict_breed(image_path))
    print()

detect_breed("images/Brittany_02625.jpg")
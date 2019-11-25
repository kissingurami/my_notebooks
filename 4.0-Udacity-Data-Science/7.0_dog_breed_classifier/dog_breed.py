from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/dogImages/train')
valid_files, valid_targets = load_dataset('data/dogImages/valid')
test_files, test_targets = load_dataset('data/dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


import cv2
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.



from keras.applications.resnet50 import ResNet50
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')


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

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

import time

human_count = 0
dog_count = 0

start_time = time.time()

for i in range(10):
    ## Test for Human faces
    if dog_detector(human_files_short[i]):
        human_count += 1
    ## Test it on dog faces next
    if dog_detector(dog_files_short[i]):
        dog_count +=1
stop_time = time.time()

duration = stop_time - start_time

print("Percentage of dog image detected in the Human Files is {}%".format(human_count))
print("Percentage of dog image detected in the Dog Files is {}%".format(dog_count))
print("Time taken in seconds for both detection algorithms on 100 samples each is : {:4.2f}".format(duration))


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pre-process the data for Keras
test_tensors = paths_to_tensor(test_files).astype('float32')/255

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model

model = Sequential()

### TODO: Define your architecture.
#LeNet based inception layer model will be used.
input_shape = (224, 224, 3)
input_img = Input(shape = input_shape)

# First Convolution Layer
conv_1 = Conv2D(4, (5,5), strides =(1,1), padding='same', activation='relu')(input_img)

# Max Pool Layer
maxpool1 = MaxPooling2D((2,2))(conv_1)

# Second Conv layer
conv_2 = Conv2D(4, (3,3), strides=(2,2), padding='same', activation='relu')(maxpool1)

# First inception Layer
path1_1 = Conv2D(8, (1,1), padding='same', activation='relu')(conv_2)
path1_1 = Conv2D(8, (3,3), padding='same', activation='relu')(path1_1)

path1_2 = Conv2D(8, (1,1), padding='same', activation='relu')(conv_2)
path1_2 = Conv2D(8, (5,5), padding='same', activation='relu')(path1_2)

path1_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(conv_2)
path1_3 = Conv2D(8, (1,1), padding='same', activation='relu')(path1_3)

inception_out1 = concatenate([path1_1, path1_2, path1_3], axis = 3)

maxpool2 = MaxPooling2D((2,2))(inception_out1)

interim_1 = Conv2D(4, (1,1), padding='same', activation='relu')(maxpool2)

fc1 = Flatten()(interim_1)

out = Dense(133, activation='softmax')(fc1)

model = Model(inputs = input_img, outputs = out)

model.summary()


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('saved_models/weights.best.from_scratch.hdf5')
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
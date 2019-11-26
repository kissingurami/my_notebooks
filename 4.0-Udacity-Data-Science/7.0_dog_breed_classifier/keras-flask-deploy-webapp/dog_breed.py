
import numpy as np
from glob import glob
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image
from tqdm import tqdm
import cv2
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

from enum import Enum

class IsDog(Enum):
    yes = 1
    no  = 2
    neithor = 3

class DogBreedDetector:

    def __init__(self):
        # load list of dog names
        self.dog_names = [item[20:-1] for item in sorted(glob("data/dogImages/train/*/"))]
        print('There are %d total dog categories.' % len(self.dog_names))

        # extract pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascades/haarcascade_frontalface_alt.xml')

        # extract pre-trained isDog detector
        # define ResNet50 model
        self.ResNet50_model = ResNet50(weights='imagenet')

        # load pretrain dog breed detector network
        self.network = 'Xception'
        input_shape = (7, 7, 2048)
        print('input_shape', input_shape)
        # define low half network
        self.Network_model = Sequential()
        self.Network_model.add(GlobalAveragePooling2D(input_shape=input_shape))
        self.Network_model.add(Dense(133, activation='softmax'))
        self.Network_model.compile(loss='categorical_crossentropy',
                              optimizer='rmsprop', metrics=['accuracy'])

        model_filepath = 'models/weights.best.' + self.network + '.hd5'
        self.Network_model.load_weights(model_filepath)
        print("The best weights for the {} model have been loaded".format(self.network))

        print("Warm up!")
        self.detect_breed("images/Brittany_02625.jpg")

    # utilities
    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self, img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    # returns "True" if face is detected in image stored at img_path
    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0


    def ResNet50_predict_labels(self, img_path):
        # returns prediction vector for image located at img_path
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(self.ResNet50_model.predict(img))

    # returns "True" if a dog is detected in the image stored at img_path
    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def network_predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = self.extract_Xception(self.path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.Network_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return self.dog_names[np.argmax(predicted_vector)]

    def detect_breed(self, image_path):
        print(image_path)
        isDog = IsDog.yes
        if self.face_detector(image_path):
            print("Hello, human!")
            isDog = IsDog.no
        elif self.dog_detector(image_path):
            print("Hello, dog!")
        else:
            isDog = IsDog.neithor
            print("Error: Neither a human face or a dog was detected.\n")
            return isDog, None

        print("You look like a ...")
        pred = self.network_predict_breed(image_path)
        print(pred)
        # img = preprocess_input(self.path_to_tensor(image_path))
        # preds = self.ResNet50_model.predict(img)
        # print('predicts:', decode_predictions(preds, top=1))
        return isDog, pred

    def extract_Resnet50(self, tensor):
        from keras.applications.resnet50 import ResNet50, preprocess_input
        return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

    def extract_Xception(self, tensor):
        from keras.applications.xception import Xception, preprocess_input
        return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

if __name__ == '__main__':
    dog_detector = DogBreedDetector()

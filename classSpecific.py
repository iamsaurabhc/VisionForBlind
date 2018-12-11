from numpy import array
import numpy as np
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import string
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax

class FeatureExtraction:
    def __init__(self, photoDir, textDir):
        self.photoDir = photoDir
        self.textDir = textDir

    # extract features from each photo in the directory
    def extractFeatures(self):
        # load the model
        model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # summarize
        print(model.summary())
        # extract features from each photo
        features = dict()
        for name in listdir(self.photoDir):
            # load an image from file
            filename = self.photoDir + '/' + name
            image = load_img(filename, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get features
            feature = model.predict(image, verbose=0)
            # get image id
            image_id = name.split('.')[0]
            # store feature
            features[image_id] = feature
            print('>%s' % name)
        return features

    def run(self):
        # extract features from all images
        features = self.extractFeatures()
        print('Extracted Features: %d' % len(features))
        # save to file
        dump(features, open('features.pkl', 'wb'))        

photoDir = '../train2014/'
textDir = '../annotations/captions_train2014.json'

FeatureExtraction(photoDir, textDir).run()
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
import os.path
import json

class FeatureExtraction:
    def __init__(self, photoDir, textDir):
        self.photoDir = photoDir
        self.annotationJson = textDir

    # extract features from each photo in the directory
    def extractPhotoFeatures(self):
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

    def extractPicFeatures(self):
        # extract features from all images
        features = self.extractPhotoFeatures()
        print('Extracted Features: %d' % len(features))
        # save to file
        dump(features, open('features.pkl', 'wb'))   
    
    # load doc into memory
    def loadAnnotationDoc(self):
        # open the file as read only
        with open(self.annotationJson, 'r') as fp:
            text = json.load(fp)

        return text


    # extract descriptions for images
    def loadDescriptions(self, doc):
        mapping = dict()
        # process lines
        for annotations in doc['annotations']:
            # take the first token as the image id, the rest as the description
            image_id, image_desc = annotations['image_id'], annotations['captions']
            # create the list if needed
            if image_id not in mapping:
                mapping[image_id] = list()
            # store description
            mapping[image_id].append(image_desc)
        return mapping

    def cleanDescriptions(self,descriptions):
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in descriptions.items():
            for i in range(len(desc_list)):
                desc = desc_list[i]
                # tokenize
                desc = desc.split()
                # convert to lower case
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [w.translate(table) for w in desc]
                # remove hanging 's' and 'a'
                desc = [word for word in desc if len(word)>1]
                # remove tokens with numbers in them
                desc = [word for word in desc if word.isalpha()]
                # store as string
                desc_list[i] =  ' '.join(desc)

    # convert the loaded descriptions into a vocabulary of words
    def toVocabulary(self, descriptions):
        # build a list of all description strings
        all_desc = set()
        for key in descriptions.keys():
            [all_desc.update(d.split()) for d in descriptions[key]]
        return all_desc    

    # save descriptions to file, one per line
    def saveDescriptions(self, descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()

    def extractTextFeatures(self):
        # load annotations
        doc = self.loadAnnotationDoc()
        # parse descriptions
        descriptions = self.loadDescriptions(doc)
        print('Loaded: %d ' % len(descriptions))
        # clean descriptions
        self.cleanDescriptions(descriptions)
        # summarize vocabulary
        vocabulary = self.toVocabulary(descriptions)
        print('Vocabulary Size: %d' % len(vocabulary))
        # save to file
        self.saveDescriptions(descriptions, 'descriptions.txt')   


    def run(self):
        #self.extractPicFeatures() # DONE
        self.extractTextFeatures()

photoDir = 'train2014'
textDir = 'annotations/captions_train2014.json'

FeatureExtraction(photoDir, textDir).run()
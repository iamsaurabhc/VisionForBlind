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
import pdb

class FeatureExtraction:
    def __init__(self, photoDir, textDir):
        self.photoDir = photoDir
        self.annotationJson = textDir
        self.tokenizer = None
        self.max_length = None
        self.vocab_size = None
        self.model = None

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
            img_name = name.split('.')[0]
            image_id = int(img_name.split('_')[2])
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
            image_id, image_desc = annotations['image_id'], annotations['caption']
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
                lines.append(str(key) + ' ' + desc)
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

    # load a pre-defined list of photo identifiers
    def loadSet(self):
        doc = self.loadAnnotationDoc()
        dataset = list()
        # process line by line
        for annotation in doc['annotations']:
            # get the image identifier
            identifier = annotation['image_id']
            dataset.append(identifier)
        return set(dataset) 
    
    def loadDocument(self, docName):
        # open the file as read only
        file_ = open(docName, 'r')
        # read all text
        text = file_.read()
        # close the file
        file_.close()
        return text

    # calculate the length of the description with the most words
    def maxLength(self, descriptions):
        lines = self.toLines(descriptions)
        return max(len(d.split()) for d in lines)

    # load clean descriptions into memory
    def loadCleanDescriptions(self, filename, dataset):
        # load document
        doc = self.loadDocument(filename)
        print('train1:5\n',list(dataset)[0:5])
        counter = 0
        descriptions = dict()
        print('doc')
        for line in doc.split('\n'):
            if counter<5:
                print(line)
                counter+=1
            # split line by white space
            tokens = line.split()
            # split id from description
            image_id, image_desc = tokens[0], tokens[1:]
            if counter<5:
                print('image_id:',image_id)
                print('image_desc:',image_desc)
            # skip images not in the set
            if image_id in dataset:
                print('id:',image_id,'desc:',image_desc)
                # create list
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # wrap description in tokens
                desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
                # store
                descriptions[image_id].append(desc)
        return descriptions
    
    # load photo features
    def loadPhotoFeatures(self, filename, dataset):
        # load all features
        all_features = load(open(filename, 'rb'))
        # filter features
        features = {k: all_features[k] for k in dataset}
        return features

    # covert a dictionary of clean descriptions to a list of descriptions
    def toLines(self, descriptions):
        all_desc = list()
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc

    # fit a self.tokenizer given caption descriptions
    def createTokenizer(self,descriptions):
        lines = self.toLines(descriptions)
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(lines)
        return self.tokenizer

    # create sequences of images, input sequences and output words for an image
    def createSequences(self, tokenizer, max_length, descriptions, photos):
        X1, X2, y = list(), list(), list()
        # walk through each image identifier
        for key, desc_list in descriptions.items():
            # walk through each description for the image
            for desc in desc_list:
                # encode the sequence
                seq = self.tokenizer.texts_to_sequences([desc])[0]
                # split one sequence into multiple X,y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    # store
                    X1.append(photos[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
        return array(X1), array(X2), array(y)

    # define the captioning model
    def defineModel(self):
        # feature extractor model
        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # summarize model
        print(model.summary())
        plot_model(model, to_file='model.png', show_shapes=True)
        return model

    def prepareTrainData(self):
        # load training dataset (83K)
        train = self.loadSet()
        print('Dataset: %d' % len(train))
        # descriptions
        train_descriptions = self.loadCleanDescriptions('descriptions.txt', train)
        print('Descriptions: train=%d' % len(train_descriptions))
        # photo features
        train_features = self.loadPhotoFeatures('features.pkl', train)
        print('Photos: train=%d' % len(train_features))
        # prepare tokenizer
        self.tokenizer = self.createTokenizer(train_descriptions)
        # save the tokenizer
        dump(self.tokenizer, open('tokenizer.pkl', 'wb'))
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % self.vocab_size)
        # determine the maximum sequence length
        self.max_length = self.maxLength(train_descriptions)
        print('Description Length: %d' % self.max_length)
        # prepare sequences
        X1train, X2train, ytrain = self.createSequences(self.tokenizer, self.max_length, train_descriptions, train_features)
        return X1train, X2train, ytrain
    
    def prepareTestData(self):
        # dev dataset
        # load test set
        self.annotationJson = 'annotations/captions_train2014.json'
        test = self.loadSet()
        print('Dataset: %d' % len(test))
        # descriptions
        test_descriptions = self.loadCleanDescriptions('descriptions.txt', test)
        print('Descriptions: test=%d' % len(test_descriptions))
        # photo features
        test_features = self.loadPhotoFeatures('features.pkl', test)
        print('Photos: test=%d' % len(test_features))
        # prepare sequences
        X1test, X2test, ytest = self.createTokenizer(self.tokenizer, self.max_length, test_descriptions, test_features)
        return X1test, X2test, ytest

    def fitModel(self):
        X1train, X2train, ytrain = self.prepareTrainData()
        X1test, X2test, ytest = self.prepareTestData()
        # fit model
        # define the model
        self.model = self.defineModel()
        # define checkpoint callback
        filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # fit model
        self.model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


    def run(self):
        self.extractPicFeatures()
        self.extractTextFeatures()
        self.fitModel()

photoDir = 'train2014'
textDir = 'annotations/captions_train2014.json'

FeatureExtraction(photoDir, textDir).run()
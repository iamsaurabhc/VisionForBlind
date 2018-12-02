from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from os import listdir
import string


class preparePhoto:
    '''
    We will use a pre-trained model to interpret the content of the photos.
    '''
    def __init__(self):
        self.features = None
    
    def extract_features(self):
        '''
        extract features from each photo in the directory
        '''
        directory = 'Flicker8k_Dataset'
        # load the model
        model = VGG16()
        # re-structure the model
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        # summarize
        print(model.summary())
        # extract features from each photo
        features = dict()
        for name in listdir(directory):
            # load an image from file
            filename = directory + '/' + name
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
        self.features = features
        print('Extracted Features: %d' % len(features))
        # save to file
        dump(features, open('features.pkl', 'wb'))

class prepareText:
    '''
    PRepare the text for deep learning model
    '''

    def __init__(self):
        self.vocabulary = None
        self.description = None

    # load doc into memory
    def load_doc(self,filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # extract descriptions for images
    def load_descriptions(self,doc):
        mapping = dict()
        # process lines
        for line in doc.split('\n'):
            # split line by white space
            tokens = line.split()
            if len(line) < 2:
                continue
            # take the first token as the image id, the rest as the description
            image_id, image_desc = tokens[0], tokens[1:]
            # remove filename from image id
            image_id = image_id.split('.')[0]
            # convert description tokens back to string
            image_desc = ' '.join(image_desc)
            # create the list if needed
            if image_id not in mapping:
                mapping[image_id] = list()
            # store description
            mapping[image_id].append(image_desc)
        return mapping

    def clean_descriptions(self):
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for key, desc_list in self.descriptions.items():
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
    def to_vocabulary(self):
        # build a list of all description strings
        all_desc = set()
        for key in self.descriptions.keys():
            [all_desc.update(d.split()) for d in self.descriptions[key]]
        return all_desc

    # save descriptions to file, one per line
    def save_descriptions(self,descriptions, filename):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()
    
    def run(self):
        filename = 'Flickr8k_text/Flickr8k.token.txt'
        # load descriptions
        doc = self.load_doc(filename)
        # parse descriptions
        self.descriptions = self.load_descriptions(doc)
        print('Loaded: %d ' % len(self.descriptions))
        # clean descriptions
        self.clean_descriptions()
        # summarize vocabulary
        self.vocabulary = self.to_vocabulary()
        print('Vocabulary Size: %d' % len(self.vocabulary))
        # save to file
        self.save_descriptions(self.descriptions, 'descriptions.txt')

class loadDataForTraining:
    

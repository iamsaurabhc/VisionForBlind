class imageData:
    '''A class to generate image specific features and dump them in intermediary file for future usages'''

    def __init__(self):
        from pickle import dump
        # Extract features from all images in directory
        directory = 'Flicker8k_Dataset'
        features = self.extractFeatures(directory)
        print "Extracted Features: %d" % len(features)
        # Save the features to a file for future reuse
        dump(features,open('features.pkl','wb'))

    def extractFeatures(self,directory):
        '''Extracts features from each image (1-dimensional 4,096 element vector) in a directory using a pre-trained model to interpret the content of an image'''
        from os import listdir
        from keras.applications.vgg16 import VGG16
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array
        from keras.applications.vgg16 import preprocess_input
        from keras.models import Model
        # Load model
        model = VGG16()
        # Restructure the model
        model.layers.pop()
        # Remove last layer which was used for classification | We only need features
        model = Model(inputs=model.inputs, outputs= model.layers[-1].output)
        # Summarize the model
        print(model.summary())
        # Extract features from each photo
        features = dict()
        for name in listdir(directory):
            # Load an image from file
            filename = directory + '/' + name
            image = load_img(filename, target_size = (224,224))
            # Convert the image pixels to a Numpy array
            image = img_to_array(image)
            # Reshape Image array data for model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # Prepare the image for the VCG model
            image = preprocess_input(image)
            # Get features from images
            feature = model.predict(image, verbose = 0)
            # Get Image Id
            image_id = name.split('.')[0]
            # Store feature in a dictionary
            features[image_id] = feature
            print('>%s' % name)
        return features

#imageData()

class textData:
    '''Class to preprocess the textual data (captions)'''
    def __init__(self):
        filename = 'Flickr8k_text/Flickr8k.token.txt'
        # load image descriptions
        doc = self.loadDocument(filename)
        # parse descriptions
        descriptions = self.loadImageDescriptions(doc)
        print('Loaded: %d image descriptions' % len(descriptions))
        # clean descriptions
        self.cleanImageDescriptions(descriptions)
        # summarize vocabulary
        vocab = self.toVocab(descriptions)
        print('Vocabulary Size: %d words' % len(vocab))
        # save to file
        self.saveImageDescription(descriptions)
    
    def loadDocument(self, filename):
        '''Load the image captions into memory'''
        # open the file as read-only
        file = open(filename,'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        # return textual contents of file
        return text
    
    def loadImageDescriptions(self, doc):
        '''Map image specific descriptions to imageIDs using a dictionary of key value pair'''
        mapping = dict()
        # process lines
        for line in doc.split('\n'):
            # split lines by white space
            tokens = line.split()
            if len(line)<2:
                continue
            # take first token as image ID and rest as description
            imgID, imgDesc = tokens[0], tokens[1:]
            # remove filename from imgID
            imgID = imgID.split('.')[0]
            # convert imgDesc back to string
            imgDesc = ' '.join(imgDesc)
            # create the dictionary for imageID
            if imgID not in mapping:
                mapping[imgID] = list()
            # store the image specific description
            mapping[imgID].append(imgDesc)
        return mapping

    def cleanImageDescriptions(self, descriptions):
        '''
        Clean the text in the following ways in order to reduce the size of the vocabulary of words we will need to work with:
            -Convert all words to lowercase.
            -Remove all punctuation.
            -Remove all words that are one character or less in length (e.g. 'a').
            -Remove all words with numbers in them.
        '''
        import string
        for key,descList in descriptions.items():
            for i in range(len(descList)):
                desc = descList[i]
                # tokenize
                desc = desc.split()
                # convert to lowercase
                desc = [word.lower() for word in desc]
                # remove punctuation from each token
                desc = [w.translate(None, string.punctuation) for w in desc]
                # remove single characters
                desc = [word for word in desc if len(word)>1]
                # remove words with numbers
                desc = [word for word in desc if word.isalpha()]
                # store as string
                descList[i] = ' '.join(desc)
    
    def toVocab(self, descriptions):
        '''Transform the clean descriptions into a set and print its size to get an idea of the size of our dataset vocabulary'''
        allDesc = set()
        for key in descriptions.keys():
            [allDesc.update(d.split()) for d in descriptions[key]]
        return allDesc

    def saveImageDescription(self, descriptions):
        '''Save the dictionary of image identifiers and descriptions to a new file named descriptions.txt, with one image identifier and description per line.'''
        filename = 'imageDescriptions.txt'
        lines = list()
        for key,descList in descriptions.items():
            for desc in descList:
                lines.append(key+ ' '+desc)
        data = '\n'.join(lines)
        files = open(filename,'w')
        files.write(data)
        files.close()

# textData()
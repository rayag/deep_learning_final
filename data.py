import numpy as np
from keras.preprocessing.image import ImageDataGenerator

'''
Structure holding model data
'''
class Data:
    def __init__(self, train_dataset, train_labels, 
                 valid_dataset, valid_labels, 
                 test_dataset, test_labels,
                 datagen=None):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.test_dataset = test_dataset
        self.test_labels = test_labels
        self.datagen = datagen
        if datagen is not None:
            datagen.fit(self.train_dataset)

    def train_size(self):
        return self.train_dataset.shape[0]
    
    def next_batch(self, batch_size):
        if self.datagen is None:
            indices = np.random.randint(self.train_size(), size=batch_size)
            yield self.train_dataset[indices], self.train_labels[indices]
        else:
            yield next(self.datagen.flow(self.train_dataset, self.train_labels, shuffle=False, batch_size=batch_size))

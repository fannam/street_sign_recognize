import tensorflow
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from keras import Model
#functional approach: function that returns a model
#this approach is good and useful for real project
def functional_model(): #flexible
    #method's parameter could be img size, channel, filter...
    my_input = Input(shape=(28,28,1)) #img: 28x28 pixel from mnist, 1 channel because of grayscaled
    #Conv2D need 4D tensor, (28,28,1) is 4D since 1 dim is hidden
    #parameter to train model, decide model's accuracy
    x = Conv2D(32, (3,3), activation='relu')(my_input) #32 filter, each one size 3x3
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x) #filter can be change to whatever
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x) #units can be chosen differently
    #output layer:
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs = my_input, outputs = x)

    return model


#keras.Model: inherit from this class
class MyCustomModel(keras.Model):

    def __init__(self)->None:
        super().__init__()

        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x

def streesigns_model(nbr_classes):

    my_input = Input(shape=(60,60,3)) #estimate the mean or medium of img's size
    #channel = 3 since RGB
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

if __name__ == '__main__':
    model = streesigns_model(10)
    model.summary()
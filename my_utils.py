import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from keras.preprocessing.image import ImageDataGenerator

def display_some_examples(examples, labels):
    
    plt.figure(figsize=(10,10))

    for i in range(25):
        idx = np.random.randint(0, examples.shape[0]-1)
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    
    plt.show()

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):

    folders = os.listdir(path_to_data)
    
    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_path = glob.glob(os.path.join(full_path, '*.png'))

        x_train, x_val = train_test_split(images_path, test_size=split_size)
        #x_train is an array of img path

        for x in x_train:
            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)
        
        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)

def order_test_set(path_to_images, path_to_csv): #classify image into folder by its label

    testset = {} #dictionary

    try:

        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')

            for i, row in enumerate(reader):
                if i==0:
                    continue #the first row contains columns' name
            
                img_name = row[-1].replace('Test/', '') #take only the image's name
                label = row[-2] #second from the last
                
                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)
    
    except:

        print('[INFO] : Error reading csv file')

    finally:
        
        return testset

def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    preprocessor = ImageDataGenerator(
        rescale = 1.0 / 255.0
    )

    train_generator = preprocessor.flow_from_directory(
        directory=train_data_path,
        target_size=(60,60),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = preprocessor.flow_from_directory(
        directory=val_data_path,
        target_size=(60,60),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    test_generator = preprocessor.flow_from_directory(
        directory=test_data_path,
        target_size=(60,60),
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator
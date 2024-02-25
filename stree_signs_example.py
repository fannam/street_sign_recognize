from sklearn.model_selection import train_test_split
from my_utils import split_data, order_test_set, create_generators
from DL_models import streesigns_model
from keras.callbacks import ModelCheckpoint, EarlyStopping 
import keras
#EarlyStopping: if after some epochs, model's accuracy is not increase then stop training



if __name__=="__main__":
    
    if False:
        path_to_data = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\Train"
        path_to_save_train = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\training_data\\train"
        path_to_save_val = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\training_data\\val"
        split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    
    if False:
        path_to_images = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\Test"
        path_to_csv = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\Test.csv"
        order_test_set(path_to_images, path_to_csv)
    
    path_to_save_train = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\training_data\\train"
    path_to_save_val = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\training_data\\val"
    path_to_test = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\Test"
    batch_size = 64 #recommend: batch_size = 2^n, not too large
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_save_train, path_to_save_val, path_to_test)
    nbr_classes = train_generator.num_classes
    TRAIN = False
    TEST = True

    if TRAIN:
        #save the best model (highest validation accuracy)
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            #if monitor="val_loss" then mode='min' to get the best model
            save_best_only=True, #save 1 model, if = False -> save many models
            save_freq='epoch',
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor="val_accuracy",
            patience=10 #if after 10 epochs, model's accuracy is not increase, stop training
            #useful with many epochs and large dataset
        )

        model = streesigns_model(nbr_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #class_mode = categorycal

        model.fit(
            train_generator, #generator contains (x_train, y_train)
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver, early_stop]
            )
    if TEST:
        model = keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)
        print("Evaluating test set:")
        model.evaluate(test_generator)

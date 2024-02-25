import tensorflow as tf
import numpy as np
import keras
def predict_with_model(model, img_path):
 
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, [60,60]) #shape = (60,60,3)
    image = tf.expand_dims(image, axis=0) #shape = (1,60,60,3) 4d tensor
    #that means 1 image size (60,60,3) 

    predictions = model.predict(image) #list of distribution (after softmax)
    classified = np.argmax(predictions)

    return classified

if __name__ == "__main__":

    img_path = "D:\\BKA\\DS, ML, DL\\German Traffic Sign Dataset\\Test\\2\\00577.png"
    model = keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")
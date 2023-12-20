
import pandas as pd
import tensorflow as tf


# reading test and train file
train_images = pd.read_csv("archive/csvTrainImages 60k x 784.csv")
train_lables = pd.read_csv("archive/csvTrainLabel 60k x 1.csv")
test_images = pd.read_csv("archive/csvTestImages 10k x 784.csv")
test_lables = pd.read_csv("archive/csvTestLabel 10k x 1.csv")



#normalization (pixils 0-255 range, lables range 0-9)
train_images = train_images / 255
test_images = test_images / 255

#reshaping dataset to pass to our model
train_images = train_images.values.reshape(-1, 784)
train_lables = train_lables.values

test_images = test_images.values.reshape(-1, 784)
test_lables = test_lables.values


def create_model():
    model = tf.keras.models.Sequential([
    
    tf.keras.layers.InputLayer(input_shape=(784,)),
    
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    
    ])
    return model

class myCallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        accuracy = logs.get('accuracy')  # Change 'acc' to 'accuracy'
        if accuracy is not None and accuracy >= 0.998:
            print("Reached 99% accuracy. Cancelling training....")
            self.model.stop_training = True
        else:
            print(f"\n\ncurrent accuracy: {accuracy}\n\n")

#neural network
def neural_net():
    mcallbacks = myCallBacks()
    
    model = create_model()
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    # model fitting
    history = model.fit(train_images, train_lables, epochs=500, callbacks=[mcallbacks], validation_data=(test_images, test_lables))
    model.save('mnis.h5')
    print("Saving the model as mnist.h5")
    return history.epoch, history.history['accuracy'][-1] 


# calling our function  
neural_net()    

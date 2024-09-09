import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv
import pandas as pd
import io
import os
import cv2
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Input 
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model
import time

#Import data
dataset = pd.read_csv('C:/Users/nikhitha/Desktop/Dataset/trainLabels.csv',on_bad_lines='skip')
image_folder_path = "C:/Users/nikhitha/Desktop/Dataset/Images"

#Convert labels to one-hot-encoded format represent categorical data as numerical vectors.
one_hot_labels = pd.Series(dataset['level'])
one_hot_labels = pd.get_dummies(one_hot_labels, sparse = True)
one_hot_labels = np.asarray(one_hot_labels)

x = []
y = []
fixed_size = (300, 300)

# resize them to a fixed size of (300, 300).
#  This fixed size is chosen to maintain retaining sufficient detail in the images for accurate classification.
for index, row in dataset.iterrows():
  image_path = os.path.join(image_folder_path, row['image']+'.jpeg')
  image = cv2.imread(image_path)
  if image is None:
    pass
  else: 
    resized_image = cv2.resize(image, fixed_size)
    x.append(resized_image)
    y.append(one_hot_labels[index])

X = x + x
Y = y + y
del x
del y
del one_hot_labels

# Split the dataset into training and validation sets into 80:20
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state=42)

#Load the pre-trained ResNet50 model
pretrained_model = tf.keras.applications.ResNet50(
    input_shape=(300, 300, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)

pretrained_model.trainable = False
inputs = pretrained_model.input

conv1 = Conv2D(32, kernel_size=4, activation='relu')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, kernel_size=4, activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(64, kernel_size=4, activation='relu')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(32, kernel_size=4, activation='relu')(pool4)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

flat = Flatten()(pool5)

hidden1 = Dense(128, activation='relu')(flat)
hidden2 = Dense(128, activation='relu')(hidden1)
hidden3 = Dense(64, activation='relu')(hidden2)
outputs = Dense(5, activation='softmax')(hidden3)

#Build a custom model on top of the pre-trained ResNet50 model with Conv2D, MaxPooling2D,  Dense, Flatten, and BatchNormalization layers.
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

#Compile the model with 'adam' optimizer, 'categorical_crossentropy' loss function, and 'accuracy' metric.
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy']) 

y_train_raw = np.asarray(y_train).astype(np.float32)
x_train_raw = np.asarray(x_train).astype(np.float32)
x_valid_raw = np.asarray(x_valid).astype(np.float32)
y_valid_raw = np.asarray(y_valid).astype(np.float32)


# Fit the model on the training data for 15 epochs with a batch size of 32 and an early stopping callback.
start_time = time.time()
history = model.fit(
    x_train_raw, 
    y_train_raw,
    validation_data = (x_valid_raw, y_valid_raw),
    batch_size = 32,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 15,
            restore_best_weights = True
        )
    ]
)
end_time = time.time()
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))     

# Evaluate the Model on the validation data and print the test loss, test accuracy, and final validation accuracy.
score = model.evaluate(x_valid_raw, y_valid_raw, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
final_val_acc_tf = history.history['val_accuracy'][-1]
print("Final validation accuracy for TensorFlow: {:.4f}".format(final_val_acc_tf))     
# Save the TensorFlow model
model.save('tensorflow_model.h5')
    
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and validation accuracy')
plt.show()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and validation loss')
plt.show()
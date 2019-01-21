import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense , Conv2D, Dropout, Flatten, MaxPooling2D
from keras.utils import np_utils
from keras import regularizers
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint
import h5py
import os
import cv2
from sklearn.model_selection import train_test_split

# loading data from the path specified
def data_loader(path_train,path_test):
   train_list=os.listdir(path_train)
   # train_list contains mapped class names to interger labels
   num_classes=len(train_list)

   # creating empty lists of training and testing dataset
   x_train=[]
   y_train=[]
   x_test=[]
   y_test=[]

   # Loading training data
   for label,elem in enumerate(train_list):

           path1=path_train+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)
               # Append image to the train data list
               x_train.append(img)
               # Append class-label corresponding to the image
               y_train.append(str(label))

           # Loading testing data
           path1=path_test+'/'+str(elem)
           images=os.listdir(path1)
           for elem2 in images:
               path2=path1+'/'+str(elem2)
               # Read the image form the directory
               img = cv2.imread(path2)
               # Append image to the test data list
               x_test.append(img)
               # Append class-label corresponding to the image
               y_test.append(str(label))

   # Converting lists into numpy arrays
   x_train=np.asarray(x_train)
   y_train=np.asarray(y_train)
   x_test=np.asarray(x_test)
   y_test=np.asarray(y_test)
   return x_train,y_train,x_test,y_test


path_train='./Data/train'
path_test='./Data/test'

X_train,y_train,X_test,y_test=data_loader(path_train,path_test)

input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
# forcing the precision of the pixel values to be 32 bit
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs between 0-1
X_train = X_train / 255.
X_test = X_test / 255.
#converting labels into One-Hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Splitting the trining data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def model():
	# create model
	model = Sequential()
	#We will add 2 Convolution layers with 32 filters of 3x3, keeping the padding as same
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same' , input_shape = input_shape, activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Conv2D(32, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	#Pooling the feature map using a 2x2 pool filter
	model.add(MaxPooling2D((2, 2), strides=(2, 2), padding = 'valid'))
	#Adding 2 more Convolutional layers having 64 filters of 3x3
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), padding = 'same', activation = 'relu', kernel_initializer = 'glorot_uniform', kernel_regularizer = regularizers.l2(0.01)))
	#Flatten the feature map
	model.add(Flatten())
	#Adding FC Layers
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.3))
	#A softmax activation function is used on the output
	#to turn the outputs into probability-like values and
	#allow one class of the 10 to be selected as the model's output #prediction.
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	#Checking the model summary
	# model.summary()
	# Loading weigths
	#model.load_weights('./CNN.h5')
	# Compile model
	# sgd = SGD(lr = 0.001, momentum = 0.0005, decay = 0.0005)
	# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0005)
	model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
	#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return model

model = model()
# Fit the model
#The model is fit over 10 epochs with updates every 200 images. The test data is used as the validation dataset
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=200, verbose=1)


# checkpoint
# filepath='./CNN.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# # Fit the model
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=200, callbacks=callbacks_list, verbose=1)
# #Saving the model
# model.save_weights('./CNN.h5')
model.save('model5.h5')
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print(scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

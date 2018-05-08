# ResNet50_HandSign.py
"""
Image classification: number sign of single hand
50 layers residual network
version of combination all codes in one
@author: Yen Tien Wu
Deep residual networks for image recognition, He et al., 2015. https://arxiv.org/pdf/1512.03385.pdf
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D # revised
from keras.models import Model, load_model
from keras.initializers import he_normal


def load_dataset():
    """
    64 x 64 x 3 images; train 1080 images; test 120 images; HDF5 files
    class 0 1 2 3 4 5; number sign of single hand
    
    Returns:
    train_set_x_orig -- train data image, 1080, dimension 64x64x3 
    train_set_y_orig -- train data class, 1080, dimension 1 
    test_set_x_orig -- test data image 120, dimension 64x64x3
    test_set_y_orig -- test data class 120, dimension 1 
    classes -- integer, number of classes
    """
    train_dataset = h5py.File('datasets/train_signs.h5', 'r') # HDF5 file
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', 'r') # HDF5 file
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # reshape Rank 1 array to matrix
    test_set_y_orig  = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))   # reshape Rank 1 array to matrix
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)] # reshape(-1) to make an array. np will look at the length of the array and remaining dimensions
    return Y
    
    
# Load Data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6) # matrix [m , 6] 6 column
Y_test = convert_to_one_hot(Y_test_orig, 6)


# Demon an image in train data
index = np.random.randint(len(Y_train_orig[0])) # pick a random integer from max/length number of train data
print("index = " + str(index) + " ; " + "class = " + str(Y_train_orig[0,index]))
plt.imshow(X_train_orig[index])


def identity_block(X, filters, windowSize):
    """
    Input activation a^[#layer]  Dimension SAME as Output activation a^[#layer+3]
    
    Arguments:
    X -- tensor, input, shape (m, n_Height_prev, n_Width_prev, n_Channel_prev)
    filters -- list of integers for 3 hidden layers, number of filters, main path filters in the CONV layers.
    windowSize -- integer, scanning windows size, kernel_size = (windowSize,windowSize), scanning window shape of the middle CONV in the main path 
     
    Returns:
    X -- tensor, output of the identity block, shape (n_Height, n_Weight, n_Channel)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value X as X_shortcut, to add back to the main path.
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding ='valid', kernel_initializer = he_normal())(X)
    #  kernel_size = (Widht, Height), strides = (stride along width, stride along height), padding = 'valid' no padding, 'same' dimension
    X = BatchNormalization(axis = 3)(X) # axis along which to normalize, axis = 3 is 'Channel' axis
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (windowSize,windowSize), strides = (1,1), padding = 'same', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding ='valid', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, filters, windowSize, s = 2):
    """
    Input activation a^[#layer]  Dimension DIFFERENT than Output activation a^[#layer+3]
    ##### output dimension [ ((n+2p-f)/stride) +1  ,  ((n+2p-f)/stride) +1 ]
    
    Arguments:
    X -- tensor, input, shape (m, n_Height_prev, n_Width_prev, n_Channel_prev)
    filters -- list of integers for 3 hidden layers, number of filters, main path filters in the CONV layers. 
    windowSize -- integer, scanning windows size, kernel_size = (windowSize,windowSize), scanning window shape of the middle CONV in the main path 
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- tensor, output of the identity block, shape (n_Height, n_Weight, n_Channel)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding ='valid', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    
    # Second component of main path 
    X = Conv2D(filters = F2, kernel_size = (windowSize,windowSize), strides = (1,1), padding = 'same', kernel_initializer = he_normal())(X) 
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding ='valid', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid', kernel_initializer = he_normal())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Architecture:
        Conv2D -> Batch_Norm -> ReLU -> Max_Pool  .....(Stage 1)
         -> Conv_Block -> ID_Block*2  .................(Stage 2)
         -> Conv_Block -> ID_Block*3  .................(Stage 3)
         -> Conv_Block -> ID_Block*5  .................(Stage 4)
         -> Conv_Block -> ID_Block*2  .................(Stage 5)
         -> AVG_Pool -> Output_Layer

    ##### output dimension [ ((n+2p-f)/stride) +1  ,  ((n+2p-f)/stride) +1 ] #####
    
    Arguments:
    input_shape -- tensor, the images of the dataset, shape (n_Height, n_Weight, n_Channel)
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor, shape (input_shape)
    X_input = Input(input_shape)
    
    # Zero Padding
    X = ZeroPadding2D(padding=(3,3))(X_input)
    
    # Stage 1
    X = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3,3), strides=(2,2))(X)

    # Stage 2
    X = convolutional_block(X, filters = [64, 64, 256],    windowSize = 3, s = 1)
    X = identity_block(X, filters = [64, 64, 256],    windowSize = 3)
    X = identity_block(X, filters = [64, 64, 256],    windowSize = 3)

    # Stage 3 
    X = convolutional_block(X, filters = [128, 128, 512],  windowSize = 3, s = 2)
    X = identity_block(X, filters = [128, 128, 512],  windowSize = 3)
    X = identity_block(X, filters = [128, 128, 512],  windowSize = 3)
    X = identity_block(X, filters = [128, 128, 512],  windowSize = 3)

    # Stage 4 
    X = convolutional_block(X, filters = [256, 256, 1024], windowSize = 3, s = 2)
    X = identity_block(X, filters = [256, 256, 1024], windowSize = 3)
    X = identity_block(X, filters = [256, 256, 1024], windowSize = 3)
    X = identity_block(X, filters = [256, 256, 1024], windowSize = 3)
    X = identity_block(X, filters = [256, 256, 1024], windowSize = 3)
    X = identity_block(X, filters = [256, 256, 1024], windowSize = 3)

    # Stage 5 
    X = convolutional_block(X, filters = [512, 512, 2048], windowSize = 3, s = 2)
    X = identity_block(X, filters = [512, 512, 2048], windowSize = 3)
    X = identity_block(X, filters = [512, 512, 2048], windowSize = 3)

    # AVG_PooL
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2))(X)

    # Output_Layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = he_normal())(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model



# Build model graph
model = ResNet50(input_shape = (64, 64, 3), classes = 6)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model 
train_history = model.fit(X_train, Y_train, epochs = 50, batch_size = 32)
#print(train_history.history.keys())

# Plot history of loss
plt.plot(train_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_history_loss'], loc='upper right')
plt.show()

# plot history of acc
plt.plot(train_history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_history_acc'], loc='upper left')
plt.show()



# Save model for future training
model.save('ResidualNet50_HE.h5') 

# Load model from previous trained model, and train again
#model = load_model('ResidualNet50.h5') 
#model.fit(X_train, Y_train, epochs = 1, batch_size = 32)

# Test modle
scores = model.evaluate(X_test, Y_test)
print ("Loss = " + str(scores[0]))
print ("Test Accuracy = " + str(scores[1]))

# Print the summary of model
model.summary()


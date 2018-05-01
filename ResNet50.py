def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL  ...(Stage 1)
    -> CONVBLOCK -> IDBLOCK*2  ................(Stage 2)
    -> CONVBLOCK -> IDBLOCK*3  ................(Stage 3)
    -> CONVBLOCK -> IDBLOCK*5  ................(Stage 4)
    -> CONVBLOCK -> IDBLOCK*2  ................(Stage 5)
    -> AVGPOOL -> TOPLAYER

    ##### output dimension [ ((n+2p-f)/stride) +1  ,  ((n+2p-f)/stride) +1 ]
    
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

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2))(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = he_normal())(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

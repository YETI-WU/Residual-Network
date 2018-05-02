def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Architecture:
        Conv2D -> Batch_Norm -> ReLU -> Max_Pool  .....(Stage 1)
         -> Conv_Block -> ID_Block*2  .................(Stage 2)
         -> Conv_Block -> ID_Block*3  .................(Stage 3)
         -> Conv_Block -> ID_Block*5  .................(Stage 4)
         -> Conv_Block -> ID_Block*2  .................(Stage 5)
         -> AVG_Pool -> Output_Layer

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

    # AVG_PooL
    X = AveragePooling2D(pool_size=(2,2), strides=(2,2))(X)

    # Output_Layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = he_normal())(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

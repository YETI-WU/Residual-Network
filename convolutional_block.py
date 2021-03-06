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
    
    

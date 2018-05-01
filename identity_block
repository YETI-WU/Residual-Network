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
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding ='valid', kernel_initializer = he_normal())(X)
    #  kernel_size = (Widht, Height), strides = (stride along width, stride along height), padding = 'valid' no padding, 'same' dimension
    X = BatchNormalization(axis = 3)(X) # axis along which to normalize, axis = 3 is 'Channel' axis
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (windowSize,windowSize), strides = (1,1), padding = 'same', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding ='valid', kernel_initializer = he_normal())(X)
    X = BatchNormalization(axis = 3)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
    
    

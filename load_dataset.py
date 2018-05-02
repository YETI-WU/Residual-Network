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

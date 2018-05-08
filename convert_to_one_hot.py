def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)] # reshape(-1) to make an array. np will look at the length of the array and remaining dimensions
    return Y
    

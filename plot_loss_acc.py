import matplotlib.pyplot as plt

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the handwritten data from the CSV file and convert it to float32 data type
data = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
# Display the first 10 rows of the data
data.head(10)
# Extract features(input data) by removing the column labeled '0'
X = data.drop('0', axis = 1)
# Extract labels(output data) from the column labeled '0'
y = data['0']
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# Split the dataset into training and testing sets
# The test_size parameter determines the proportion of the dataset to include in the test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Reshape the training and testing data to match the input shape expected by the model(28x28)
# The.values attribute converts the DataFrame to a NumPy array
# The new shape is(number of samples, height, width)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))
# Print the shape of the training and testing data to verify
print("Shape of Training data: ", x_train.shape)
print("Shape of Testing data: ", x_test.shape)
# Import necessary libraries
import cv2
# Shuffle the training data to visualize random samples
shuffle_data = shuffle(x_train)
# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize = (10, 10))
axes = axes.flatten()
# Iterate over the shuffled data and plot each sample
for i in range(9) :
 # Apply thresholding to convert the image to binary
 _, shu = cv2.threshold(shuffle_data[i], 30, 200, cv2.THRESH_BINARY)
 # Reshape the image to 28x28 and plot it on the subplot
 axes[i].imshow(np.reshape(shuffle_data[i], (28, 28)), cmap = "Greys")
 # Show the plot
plt.show()
 # Reshape training data to include a single channel
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
 # Reshape testing data to include a single channel
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
 # Print the new shape of the training and testing data
print("New shape of training data: ", x_train.shape)
print("New shape of testing data: ", x_test.shape)
import tensorflow 
from tensorflow.keras.utils import to_categorical
 # Convert training labels to one - hot encoded format
y_training = to_categorical(y_train, num_classes = 26, dtype = 'int')
 # Convert testing labels to one - hot encoded format
y_testing = to_categorical(y_test, num_classes = 26, dtype = 'int')
 # Print the new shape of the training and testing labels
print("New shape of training labels: ", y_training.shape)
print("New shape of testing labels: ", y_testing.shape)
 # Sequential model for stacking layers
from tensorflow.keras.models import Sequential
 # Different layers for building the model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
 # Optimizers for compiling the model
from tensorflow.keras.optimizers import SGD, Adam
 # Callbacks for training control
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
 # Creating a Sequential model
model = Sequential()
 # Adding the first convolutional layer with 64 filters, each of size 3x3, using ReLU activation
 # The input shape is set to(28, 28, 1) for 28x28 grayscale images
model.add(Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
 # Adding a max pooling layer with pool size 2x2
model.add(MaxPool2D(2, 2))
 # Adding the second convolutional layer with 64 filters, each of size 3x3, using ReLU activation
model.add(Conv2D(64, (3, 3), activation = 'relu'))
 # Adding another max pooling layer
model.add(MaxPool2D(2, 2))
 # Adding the third convolutional layer with 64 filters, each of size 3x3, using ReLU activation
model.add(Conv2D(64, (3, 3), activation = 'relu'))
 # Adding another max pooling layer
model.add(MaxPool2D(2, 2))
 # Flattening the output from convolutional layers
model.add(Flatten())
 # Adding a fully connected layer with 128 neurons using ReLU activation
model.add(Dense(128, activation = "relu"))
 # Adding another fully connected layer with 256 neurons using ReLU activation
model.add(Dense(256, activation = "relu"))
 # Adding the output layer with 26 neurons(for 26 classes) using softmax activation
model.add(Dense(26, activation = "softmax"))
 # Compile the model
model.compile(optimizer = Adam(learning_rate = 0.001), # Specify Adam optimizer with learning rate 0.001
loss = 'categorical_crossentropy', # Use categorical crossentropy as the loss function
metrics = ['accuracy']) # Monitor accuracy during training
 # Train the model
history = model.fit(x_train, # Training data
y_training, # Training labels
epochs = 5, # Number of epochs
validation_data = (x_test, y_testing)) # Validation data for monitoring performance
 # Display model summary
model.summary()
 # Save the trained model
model.save(r'handwritten_character_recog_model.h5')
 # Define mapping of class indices to characters
words = { 0:'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 8 : 
'I', 9 : 'J', 10 : 'K', 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 
: 'R', 18 : 'S', 19 : 'T', 20 : 'U', 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z' }
 # Create subplots for displaying predictions
fig, axes = plt.subplots(3, 3, figsize = (8, 9))
axes = axes.flatten()
 # Display images with predictions
for i, ax in enumerate(axes) :
 image = np.reshape(x_test[i], (28, 28))
 ax.imshow(image, cmap = "Greys")
 pred = words[np.argmax(y_testing[i])] # Get the predicted character
 ax.set_title("Prediction: " + pred) # Set title with prediction
 ax.grid() # Add grid lines
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
 # Load the pre - trained model
model = load_model('model_hand.h5')
 # Define the mapping of class indices to characters
words = { 0:'A', 1 : 'B', 2 : 'C', 3 : 'D', 4 : 'E', 5 : 'F', 6 : 'G', 7 : 'H', 
8 : 'I', 9 : 'J', 10 : 'K',
 11 : 'L', 12 : 'M', 13 : 'N', 14 : 'O', 15 : 'P', 16 : 'Q', 17 : 'R', 
18 : 'S', 19 : 'T', 20 : 'U',
 21 : 'V', 22 : 'W', 23 : 'X', 24 : 'Y', 25 : 'Z' }
 # Load and preprocess the image
image = cv2.imread('t.png')
 # Make a copy of the original image
image_copy = image.copy()
 # Convert the image to RGB color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 # Resize the image for better visualization
image = cv2.resize(image, (400, 440))
 # Apply Gaussian blur to the copy of the image
image_copy = cv2.GaussianBlur(image_copy, (7, 7), 0)
 # Convert the blurred image to grayscale
gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
 # Apply thresholding to the grayscale image
_, img_thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
 # Resize the thresholded image to match the input size of the model
final_image = cv2.resize(img_thresh, (28, 28))
final_image = np.reshape(final_image, (1, 28, 28, 1))
 # Perform prediction
prediction = words[np.argmax(model.predict(final_image))]
 # Add prediction text to the image
plt.imshow(image)
plt.text(390, 10, "PREDICTION: " + prediction, fontsize = 17, color = 'orange', 
fontweight = 'bold', verticalalignment = 'top')
plt.axis('off')
plt.show()

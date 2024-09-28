import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def preprocess_data(X):
    lh = X[:,:,:63].reshape(X.shape[0], X.shape[1], 21, 3)
    rh = X[:,:,63:].reshape(X.shape[0], X.shape[1], 21, 3)

    lh = lh - lh.mean(2)[:,:,np.newaxis,:]
    rh = rh - rh.mean(2)[:,:,np.newaxis,:]

    lh = lh.reshape(X.shape[0], X.shape[1], 63)
    rh = rh.reshape(X.shape[0], X.shape[1], 63)
    
    X_output = np.concatenate([lh, rh], axis=-1)
    return X_output

def flip_data(X, y):
    lh = X[:,:,:63].reshape(X.shape[0], X.shape[1], 21, 3)
    rh = X[:,:,63:].reshape(X.shape[0], X.shape[1], 21, 3)

    lh[:,:,:,0] = -lh[:,:,:,0]
    rh[:,:,:,0] = -rh[:,:,:,0]

    lh = lh.reshape(X.shape[0], X.shape[1], 63)
    rh = rh.reshape(X.shape[0], X.shape[1], 63)

    X_augment = np.concatenate([lh, rh], axis=-1)

    X_output = np.concatenate([X, X_augment], axis=0)
    y_output = np.concatenate([y, y], axis=0)

    return X_output, y_output

def jitter_augmentation(X, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def find_highest_prob_samples(model, evaluation_components):
    X, file_list, label_to_text = evaluation_components
    predictions = model.predict(X)
    highest_prob_samples = {}

    for i, pred in enumerate(predictions):
        class_index = np.argmax(pred)
        class_name = label_to_text[class_index]
        probability = pred[class_index]

        if class_name not in highest_prob_samples or probability > highest_prob_samples[class_name][1]:
            highest_prob_samples[class_name] = (file_list[i][0], probability)

    # Remove probabilities from highest_prob_samples
    for key, value in highest_prob_samples.items():
        highest_prob_samples[key] = value[0]

    # save the highest probability samples to a file
    with open('highest_prob_samples.json', 'w') as f:
        json.dump(highest_prob_samples, f)

def read_and_preprocess_data(data_path):
    # Initialize lists to store data and labels
    labels = []
    file_list = []
    data = []

    # Read files and create labels
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            labels.append(label)
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                file_list.append((file_path, label))
                # Load numpy array from file
                data.append(np.load(file_path))

    # Preprocess data
    X = np.array(data)[:,10:20,-126:]
    X = preprocess_data(X)
    y = np.array([labels.index(item[1]) for item in file_list])
    y = to_categorical(y)

    # Add flip data
    X, y = flip_data(X, y)

    # Create label to text dictionary
    label_to_text = {i: label for i, label in enumerate(labels)}
    with open('label_to_text.json', 'w') as f:
        json.dump(label_to_text, f)


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    evaluation_components =[X, file_list*2, label_to_text]

    return X_train, X_test, y_train, y_test, evaluation_components

def get_model(label_to_text):
    # Define the model
    model = Sequential()
    model.add(LSTM(16, return_sequences=False, activation='relu', input_shape=(10,126)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(len(label_to_text), activation='softmax'))

    # Compile the model
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

class AugmentedDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Apply jitter augmentation
        augmented_X = jitter_augmentation(batch_X)
        
        return augmented_X, batch_y

# Specify the path to the data
data_path = '../data'

# Read and preprocess data
X_train, X_test, y_train, y_test, evaluation_components = read_and_preprocess_data(data_path)
# Create data generators
train_generator = AugmentedDataGenerator(X_train, y_train, batch_size=16)
val_generator = AugmentedDataGenerator(X_test, y_test, batch_size=16)

# Define the model
model = get_model(evaluation_components[2])

# Define the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_categorical_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

# Train and evaluate the model with the checkpoint callback
model.fit(train_generator, 
          epochs=100, 
          callbacks=[checkpoint],
          validation_data=val_generator,
          validation_freq=1)

# Final evaluation
test_loss, test_accuracy = model.evaluate(val_generator)
print("--------------------------------")
print(f"Test accuracy: {test_accuracy:.4f}")

find_highest_prob_samples(model, evaluation_components)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from AiSignaler import AISignal
from OandaForex import *
import pandas as pd
import numpy as np
import random
import os

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Fetch data from Oanda
market_analysis = MarketAnalysisAndOrder(instrument='GBP_USD')
normalized_df = market_analysis.normalized_data.drop(['TIMESTAMP', 'VOLUME'], axis=1)


# Filter the DataFrame based on the 'TARGET' value
negative_targets = normalized_df[normalized_df['TARGET'] == -1]
positive_targets = normalized_df[normalized_df['TARGET'] == 1]
neutral_targets = normalized_df[normalized_df['TARGET'] == 0]

# Randomly select 50 instances from each filtered DataFrame
random.seed(42)  # Set a random seed for reproducibility

selected_negative = resample(negative_targets, replace=True, n_samples=200, random_state=42)
selected_positive = resample(positive_targets, replace=True, n_samples=200, random_state=42)
selected_neutral = resample(neutral_targets, replace=True, n_samples=2000, random_state=42)

# Concatenate the selected instances into a new DataFrame  
selected_data = pd.concat([selected_negative, selected_neutral], ignore_index=True)

# Shuffle the rows of the new DataFrame
normalized_df = selected_data.sample(frac=1, random_state=42).reset_index(drop=True)
# 'BOLLINGER_LOWER' 'BOLLINGER_UPPER'
features = normalized_df.drop(['TARGET', 'AVG_VOLUME', 'BOLLINGER_MIDDLE', 'OPEN', 'HIGH'], axis=1).values
target = normalized_df['TARGET'].values

# Split the scaled and randomized data into training and testing sets
train_size = 0.8  # Adjust the train size as needed
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=train_size, random_state=42)

# Define the sequence length (number of timesteps)
sequence_length = 30

# Create sequences for the training set
train_sequences = create_sequences(X_train, sequence_length)
train_targets = y_train[sequence_length-1:]

# Create sequences for the testing set
test_sequences = create_sequences(X_test, sequence_length)
test_targets = y_test[sequence_length-1:]

input_shape = train_sequences.shape[1:]  # Exclude the batch dimension

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, "Model")
model_path = os.path.join(model_dir, "trained_model (test).h5")
if os.path.exists(model_path):
    # Load the existing model
    model = AISignal()
    model.load_model("trained_model (test).h5")
    print("Existing model loaded.")
else:
    # Create a new model with the new configurations
    hidden_layers_config = [
    {'units': 128, 'activation': 'tanh', 'return_sequences': True},
    {'units': 64, 'activation': 'relu', 'return_sequences': True},
    {'units': 64, 'activation': 'tanh', 'return_sequences': True},
    {'units': 32, 'activation': 'sigmoid', 'return_sequences': True},
    {'units': 32, 'activation': 'tanh'}]
    model = AISignal(input_shape=input_shape, hidden_layers_config=hidden_layers_config, output_activation='tanh')
    print("New model created.")
# Train the model
model.train(train_sequences, train_targets)

# Evaluate the model
loss, accuracy = model.evaluate(test_sequences, test_targets)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Save the trained model
model.save_model("trained_model (test).h5")
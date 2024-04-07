from sklearn.utils import resample
import matplotlib.pyplot as plt
from AiSignaler import AISignal
from OandaForex import *
import pandas as pd
import numpy as np
import random

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i+sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

model = AISignal()

# Load the trained model (if needed)
model.load_model("trained_model (test).h5")

# Fetch data from Oanda
market_analysis = MarketAnalysisAndOrder(instrument='EUR_USD')
normalized_df = market_analysis.normalized_data
# Filter the DataFrame based on the 'TARGET' value
negative_targets = normalized_df[normalized_df['TARGET'] == -1]
positive_targets = normalized_df[normalized_df['TARGET'] == 1]
neutral_targets = normalized_df[normalized_df['TARGET'] == 0]

# Randomly select 50 instances from each filtered DataFrame
random.seed(42)  # Set a random seed for reproducibility

selected_negative = resample(negative_targets, replace=True, n_samples=350, random_state=42)
selected_positive = resample(positive_targets, replace=True, n_samples=50, random_state=42)
selected_neutral = resample(neutral_targets, replace=True, n_samples=50, random_state=42)

# Concatenate the selected instances into a new DataFrame 
selected_data = pd.concat([selected_negative, selected_positive, selected_neutral], ignore_index=True)

# Shuffle the rows of the new DataFrame
normalized_df = selected_data.sample(frac=1, random_state=42).reset_index(drop=True)
#,'BOLLINGER_LOWER', 'BOLLINGER_UPPER'
features = normalized_df.drop(['TARGET', 'TIMESTAMP', 'AVG_VOLUME','BOLLINGER_MIDDLE', 'OPEN', 'HIGH'], axis=1).values

# Define the sequence length (number of timesteps)
sequence_length = 30

prediction_sequences = create_sequences(features, sequence_length)

# Check the shape of prediction_sequences
print("Shape of prediction_sequences:", prediction_sequences.shape)

# Reshape prediction_sequences if needed
expected_shape = (len(prediction_sequences), sequence_length, features.shape[1])
if prediction_sequences.shape != expected_shape:
    prediction_sequences = prediction_sequences.reshape(expected_shape)

predicted_classes = model.get_signal(prediction_sequences)
target = normalized_df['TARGET'].values
trimmed_target = target[:len(predicted_classes)]

predictions_df = pd.DataFrame({
    'Predictions': predicted_classes,
    'Target': trimmed_target
})

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the predictions and target values
ax.plot(predictions_df['Predictions'], label='Predictions')
ax.plot(predictions_df['Target'], label='Target')

# Set the x-axis label
ax.set_xlabel('Time')

# Set the y-axis label and tick labels
ax.set_ylabel('Signal')
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(['Sell', 'Hold', 'Buy'])

# Add a legend
ax.legend()

# Add a title
ax.set_title('Predictions vs Target')

# Rotate the x-axis tick labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.show()
# Map predicted class indices to signal labels
'''

# Print the predicted signals
for signal in predicted_signals:
    print("Predicted Signal:", signal)'''
class ForexTradingEnv:
    def __init__(self, data, initial_balance):
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_step = 0
        self.current_position = None

    def reset(self):
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.current_position = None
        return self._get_observation()

    def step(self, action):
        # Execute the action and update the environment
        # Calculate the reward based on the profit/loss
        # Update the current_position and current_balance
        # Increment the current_step
        # Return the next observation, reward, done flag, and info
        pass
    def _get_observation(self):
        # Return the current market state and agent's position/balance
        pass
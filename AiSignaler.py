# For AISignaler
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.initializers import Orthogonal
from tensorflow.keras.optimizers import Adam
# Exception Handling Custom
from CustomException import *

# AI Signaler class (Draft; Version 0.0.1 09/01/2023)
class AISignal:
    def __init__(self, input_shape=None, hidden_layers_config=None, output_activation='sigmoid'):
        # Only build a new model if input_shape is provided
        if input_shape is not None:
            if hidden_layers_config is None:
                hidden_layers_config = [{'units': 50, 'return_sequences': True}, {'units': 50}]
            
            self.model = tf.keras.Sequential()
            
            # Add an Input layer to specify the input shape
            self.model.add(tf.keras.layers.Input(shape=input_shape))
            
            # Add LSTM layers
            for layer_config in hidden_layers_config:
                self.model.add(tf.keras.layers.LSTM(**layer_config))
            
            custom_learning_rate = 0.0001
            gradient_clip_norm = 1.0
            optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate, clipnorm=gradient_clip_norm)
            
            # Add a Dense layer for output
            self.model.add(tf.keras.layers.Dense(1, activation=output_activation))
            
            self.model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
        else:
            self.model = None  # This will be set when loading a model
    
    def _handle_exception(self, e, line_info=""):
        logging.error(f"Caught an exception [ LSTM AI ]: {e} {line_info}")

    def evaluate(self, X_test, y_test):
        # Evaluate the model on the test data
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
        return loss, accuracy
    
    def summary(self):
        self.model.summary()

    def get_model_file_path(self, file_name):
        """
        Returns the full file path for the specified model file name.
        Args:
            file_name (str): The name of the model file.
        Returns:
            str: The full file path for the specified model file.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "Model")
        file_path = os.path.join(model_dir, file_name)
        return file_path
    
    def save_model(self, file_name):
        """
        Save the model to the specified file path.
        :param file_path: String, path to save the model file.
        """
        # Get the full path of the current script
        current_file_path = os.path.abspath(__file__)

        # If you just want the directory containing the script
        current_dir = os.path.dirname(current_file_path)
        try:
            file_path = os.path.join(current_dir, "Model", file_name)
            self.model.save(file_path, save_format='tf', include_optimizer=False)
            print(f"Model successfully saved to {file_path}")
        except Exception as e:
            self._handle_exception(e, "Error in save_model")
    def load_model(self, file_name):
        """
        Load the model from the specified file path.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "Model")
        file_path = os.path.join(model_dir, file_name)

        try:
            self.model = tf.keras.models.load_model(file_path)
            custom_learning_rate = 0.0001
            optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)
            self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            print(f"Model successfully loaded from {file_path}")
        except Exception as e:
            self._handle_exception(e, "Error in load_model")
            
    def train(self, X_train, y_train):
        # Train your model here
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)
        
    def get_signal(self, X):
        # Check if the input data has the expected shape
        expected_shape = self.model.input_shape
        if X.shape[1:] != expected_shape[1:]:
            raise ValueError(f"Input shape {X.shape[1:]} does not match the expected shape {expected_shape[1:]}")

        # Make predictions
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    def __str__(self) -> str:
        return f"{self}"
    
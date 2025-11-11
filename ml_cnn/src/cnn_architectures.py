import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


class CNNArchitectures:
    """
    Class containing multiple CNN architectures for trend prediction.
    """
    
    @staticmethod
    def simple_cnn(input_shape: Tuple[int, ...], num_classes: int = 1) -> tf.keras.Model:
        """
        Simple CNN architecture for trend prediction.
        Works with small sequence lengths (e.g., 10) by using padding='same'.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes (1 for binary trend prediction)
            
        Returns:
            Keras Model with simple CNN architecture
        """
        model = models.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        
        return model
    
    @staticmethod
    def deeper_cnn(input_shape: Tuple[int, ...], num_classes: int = 1) -> tf.keras.Model:
        """
        Deeper CNN architecture for trend prediction.
        Works with small sequence lengths (e.g., 10) by using padding='same'.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes (1 for binary trend prediction)
            
        Returns:
            Keras Model with deeper CNN architecture
        """
        model = models.Sequential([
            layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.25),
            
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        
        return model
    
    @staticmethod
    def cnn_with_lstm(input_shape: Tuple[int, ...], num_classes: int = 1) -> tf.keras.Model:
        """
        CNN architecture combined with LSTM for trend prediction.
        Works with small sequence lengths (e.g., 10) by using padding='same'.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes (1 for binary trend prediction)
            
        Returns:
            Keras Model with CNN-LSTM architecture
        """
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.25),
            
            layers.LSTM(50, return_sequences=True),
            layers.Dropout(0.25),
            layers.LSTM(50),
            layers.Dropout(0.25),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
        ])
        
        return model

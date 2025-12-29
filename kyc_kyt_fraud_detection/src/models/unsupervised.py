"""
Unsupervised Models Module

Unsupervised anomaly detection models (Autoencoder).
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Autoencoder model will not work.")

from .base import BaseDefaultModel

logger = logging.getLogger("kyc_kyt.models.unsupervised")


class AutoencoderAnomalyModel(BaseDefaultModel):
    """
    Autoencoder-based anomaly detection for loan default.

    Trains on normal (non-default) loans and detects defaults as anomalies.
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        n_layers: int = 1,
        dropout: float = 0.247,
        l2_reg: float = 1.37e-06,
        learning_rate: float = 0.0066,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        use_tuned_params: bool = True
    ):
        """
        Initialize Autoencoder model.

        Parameters
        ----------
        encoding_dim : int
            Dimension of encoding layer
        n_layers : int
            Number of hidden layers
        dropout : float
            Dropout rate
        l2_reg : float
            L2 regularization strength
        learning_rate : float
            Learning rate for Adam optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs
        patience : int
            Early stopping patience
        use_tuned_params : bool
            Whether to use Optuna-tuned parameters
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder model")

        super().__init__(model_name="autoencoder", model_type="unsupervised")

        # Use tuned parameters from notebook if requested
        if use_tuned_params:
            self.encoding_dim = 8
            self.n_layers = 1
            self.dropout = 0.247
            self.l2_reg = 1.37e-06
            self.learning_rate = 0.00659
            self.batch_size = 32
        else:
            self.encoding_dim = encoding_dim
            self.n_layers = n_layers
            self.dropout = dropout
            self.l2_reg = l2_reg
            self.learning_rate = learning_rate
            self.batch_size = batch_size

        self.epochs = epochs
        self.patience = patience
        self.threshold = None
        self.model = None

    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build autoencoder architecture.

        Parameters
        ----------
        input_dim : int
            Input dimension

        Returns
        -------
        keras.Model
            Compiled autoencoder model
        """
        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        # Encoder
        sizes = []
        curr = input_dim
        for _ in range(self.n_layers):
            next_s = max(self.encoding_dim, curr // 2)
            sizes.append(next_s)
            x = layers.Dense(
                next_s,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)
            curr = next_s

        # Bottleneck
        x = layers.Dense(self.encoding_dim, activation='relu')(x)

        # Decoder
        for s in reversed(sizes):
            x = layers.Dense(
                s,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            )(x)
            if self.dropout > 0:
                x = layers.Dropout(self.dropout)(x)

        # Output
        outputs = layers.Dense(input_dim, activation='linear')(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss='mse'
        )

        return model

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        train_on_normal_only: bool = True
    ) -> "AutoencoderAnomalyModel":
        """
        Fit the autoencoder on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series, optional
            Training labels (used to filter normal samples if requested)
        train_on_normal_only : bool
            Whether to train only on normal (non-default) samples

        Returns
        -------
        AutoencoderAnomalyModel
            Fitted model (self)
        """
        logger.info("Training Autoencoder model...")

        self.feature_names = list(X.columns)

        # Filter to normal samples if requested
        if train_on_normal_only and y is not None:
            X_train = X[y == 0].values
            logger.info(
                f"Training on {len(X_train)} normal samples "
                f"({(y == 1).sum()} defaults excluded)"
            )
        else:
            X_train = X.values

        # Build model
        input_dim = X.shape[1]
        self.model = self._build_model(input_dim)

        # Train
        start_time = time.time()

        self.model.fit(
            X_train,
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                EarlyStopping(
                    patience=self.patience,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )

        self.training_time = time.time() - start_time

        # Calculate threshold on training data
        train_recon = self.model.predict(X_train, verbose=0)
        train_errors = np.mean(np.power(X_train - train_recon, 2), axis=1)
        self.threshold = np.percentile(train_errors, 95)

        self.is_fitted = True

        logger.info(
            f"Autoencoder trained in {self.training_time:.2f} seconds"
        )
        logger.info(f"Reconstruction threshold: {self.threshold:.6f}")

        return self

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Make predictions (1 = anomaly/default, 0 = normal).

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        threshold : float, optional
            Custom threshold (uses fitted threshold if None)

        Returns
        -------
        np.ndarray
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        errors = self._calculate_errors(X)
        threshold = threshold if threshold is not None else self.threshold

        return (errors > threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default based on reconstruction error.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction

        Returns
        -------
        np.ndarray
            Default probabilities (normalized reconstruction errors)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        errors = self._calculate_errors(X)

        # Normalize errors to [0, 1]
        errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

        return errors_norm

    def _calculate_errors(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate reconstruction errors.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        np.ndarray
            Reconstruction errors
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        reconstruction = self.model.predict(X_array, verbose=0)
        errors = np.mean(np.power(X_array - reconstruction, 2), axis=1)
        return errors

    def get_reconstruction_error_distribution(
        self,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get statistics of reconstruction errors.

        Parameters
        ----------
        X : pd.DataFrame
            Features

        Returns
        -------
        Dict[str, float]
            Error statistics
        """
        errors = self._calculate_errors(X)

        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'median': float(np.median(errors)),
            'threshold': float(self.threshold) if self.threshold else None
        }

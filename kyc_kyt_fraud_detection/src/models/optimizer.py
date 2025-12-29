"""
Hyperparameter Optimization Module

Uses Optuna for Bayesian hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import logging

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger = logging.getLogger("kyc_kyt.models.optimizer")
    logger.warning("Optuna not available. Hyperparameter optimization will not work.")

from .supervised import XGBoostDefaultModel, RandomForestDefaultModel

logger = logging.getLogger("kyc_kyt.models.optimizer")


class OptunaOptimizer:
    """
    Hyperparameter optimizer using Optuna with TPE sampler.

    Performs Bayesian optimization with cross-validation.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        metric: str = 'average_precision',
        direction: str = 'maximize'
    ):
        """
        Initialize Optuna optimizer.

        Parameters
        ----------
        n_trials : int
            Number of optimization trials
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed
        metric : str
            Metric to optimize ('average_precision', 'roc_auc', 'f1')
        direction : str
            Optimization direction ('maximize' or 'minimize')
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")

        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.metric = metric
        self.direction = direction
        self.study = None
        self.best_params = None

    def optimize_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_pos_weight_range: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        scale_pos_weight_range : tuple, optional
            Range for scale_pos_weight (min, max)

        Returns
        -------
        Dict[str, Any]
            Best hyperparameters
        """
        logger.info(f"Optimizing XGBoost ({self.n_trials} trials)...")

        # Calculate default scale_pos_weight
        default_spw = (y == 0).sum() / (y == 1).sum()

        if scale_pos_weight_range is None:
            scale_pos_weight_range = (1, default_spw * 2)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float(
                    'learning_rate', 0.01, 0.3, log=True
                ),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
                'scale_pos_weight': trial.suggest_float(
                    'scale_pos_weight',
                    scale_pos_weight_range[0],
                    scale_pos_weight_range[1]
                ),
                'random_state': self.random_state,
                'eval_metric': 'aucpr',
                'use_label_encoder': False
            }

            return self._cross_validate(XGBoostDefaultModel, params, X, y)

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params

        logger.info(f"Best {self.metric}: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def optimize_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels

        Returns
        -------
        Dict[str, Any]
            Best hyperparameters
        """
        logger.info(f"Optimizing Random Forest ({self.n_trials} trials)...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical(
                    'max_features', ['sqrt', 'log2', None]
                ),
                'class_weight': trial.suggest_categorical(
                    'class_weight', ['balanced', 'balanced_subsample', None]
                ),
                'random_state': self.random_state,
                'n_jobs': -1
            }

            return self._cross_validate(RandomForestDefaultModel, params, X, y)

        self.study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.random_state)
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        self.best_params = self.study.best_params

        logger.info(f"Best {self.metric}: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params

    def _cross_validate(
        self,
        model_class,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """
        Perform cross-validation with given parameters.

        Parameters
        ----------
        model_class
            Model class to use
        params : Dict[str, Any]
            Model parameters
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels

        Returns
        -------
        float
            Mean CV score
        """
        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )

        scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]

            # Create and train model
            model = model_class(params=params, use_tuned_params=False)
            model.fit(X_train_cv, y_train_cv)

            # Predict
            proba = model.predict_proba(X_val_cv)

            # Calculate score
            if self.metric == 'average_precision':
                score = average_precision_score(y_val_cv, proba)
            elif self.metric == 'roc_auc':
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y_val_cv, proba)
            elif self.metric == 'f1':
                from sklearn.metrics import f1_score
                pred = (proba >= 0.5).astype(int)
                score = f1_score(y_val_cv, pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            scores.append(score)

        return np.mean(scores)

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns
        -------
        pd.DataFrame
            History of trials
        """
        if self.study is None:
            return pd.DataFrame()

        trials = []
        for trial in self.study.trials:
            trial_dict = {
                'trial': trial.number,
                'value': trial.value,
                'state': trial.state.name
            }
            trial_dict.update(trial.params)
            trials.append(trial_dict)

        return pd.DataFrame(trials)

    def get_best_params_formatted(self) -> str:
        """
        Get best parameters formatted for easy copy-paste.

        Returns
        -------
        str
            Formatted parameters
        """
        if self.best_params is None:
            return "No optimization performed yet"

        lines = ["Best Parameters:"]
        lines.append("{")
        for k, v in self.best_params.items():
            if isinstance(v, float):
                lines.append(f"    '{k}': {v:.6f},")
            else:
                lines.append(f"    '{k}': {repr(v)},")
        lines.append("}")

        return "\n".join(lines)


def quick_optimize(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Quick optimization convenience function.

    Parameters
    ----------
    model_type : str
        'xgboost' or 'random_forest'
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training labels
    n_trials : int
        Number of optimization trials

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters
    """
    optimizer = OptunaOptimizer(n_trials=n_trials)

    if model_type == 'xgboost':
        return optimizer.optimize_xgboost(X, y)
    elif model_type == 'random_forest':
        return optimizer.optimize_random_forest(X, y)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

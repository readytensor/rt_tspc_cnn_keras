import joblib
from typing import Tuple
from preprocessing.custom_transformers import PADDING_VALUE
from schema.data_schema import TSAnnotationSchema
from sklearn.metrics import f1_score
from multiprocessing import cpu_count
from sklearn.exceptions import NotFittedError
from keras import callbacks
from keras import optimizers
from keras import losses
from keras import layers
from keras import Sequential
import keras
import tensorflow as tf
import numpy as np
import os
import random
import warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "1"

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")

device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"

print("device used: ", device)


def control_randomness(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TSAnnotator:
    """CNN Timeseries Annotator.

    This class provides a consistent interface that can be used with other
    TSAnnotator models.
    """

    MODEL_NAME = "CNN_Timeseries_Annotator"

    def __init__(
        self,
        data_schema: TSAnnotationSchema,
        encode_len: int,
        activation: str = "relu",
        lr: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 64,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Construct a new CNN TSAnnotator.

        Args:
            data_schema (TSAnnotationSchema): The data schema.
            encode_len (int): The length of the window sample.
            activation (str): The activation function.
            lr (float): The learning rate.
            max_epochs (int): The maximum number of epochs.
            batch_size (int): The batch size.
            random_state (int): random state number for reproducibility.
        """
        self.data_schema = data_schema
        self.encode_len = int(encode_len)
        self.activation = activation
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.net = self.build_NNet_model()
        self._is_trained = False
        self.random_state = random_state
        self.kwargs = kwargs

        control_randomness(self.random_state)

    def build_NNet_model(self):
        model = Sequential(name="CNN_Timeseries_Annotator")


        model.add(layers.Conv1D(
            filters=1024,
            kernel_size=3,
            strides=1,
            padding="same"
        ))

        model.add(layers.Permute((2, 1)))
        model.add(layers.Activation(self.activation))
        model.add(layers.Flatten())
        model.add(layers.Dense(self.encode_len * len(self.data_schema.target_classes)))
        model.add(layers.Reshape((self.encode_len, len(self.data_schema.target_classes))))
        model.add(layers.Softmax(axis=-1))

        model.compile(optimizer=optimizers.Adam(learning_rate=self.lr),
                      loss=losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        return model

    def _get_X_and_y(
        self, data: np.ndarray, is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        if is_train:
            if T != self.encode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len}"
                    f" length on axis 1. Found length {T}"
                )
            # we excluded the first 2 dimensions (id, time) and the last dimension (target)
            X = data[:, :, 2:-1]  # shape = [N, T, D]
            y = data[:, :, -1].astype(int)  # shape = [N, T]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, :, 2:]
            y = data[:, :, 0:2]
        return X, y

    def fit(self, train_data):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        callbacks_list = [callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                          callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=3)]

        self.net.fit(
            train_X,
            train_y,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            # shuffle=False,
            callbacks=callbacks_list,
            verbose=1,
        )

        self._is_trained = True
        return self.net

    def predict(self, data):

        X, window_ids = self._get_X_and_y(data, is_train=False)
        preds = self.net.predict(X)
        for i in range(len(preds)):
            if preds[i].shape[1] > len(self.data_schema.target_classes):
                preds[i] = preds[i][:, :-1]
        preds = np.array(preds)
        prob_dict = {}

        for index, prediction in enumerate(preds):
            series_id = window_ids[index][0][0]
            for step_index, step in enumerate(prediction):
                step_id = window_ids[index][step_index][1]
                step_id = (series_id, step_id)
                prob_dict[step_id] = prob_dict.get(step_id, []) + [step]

        prob_dict = {
            k: np.mean(np.array(v), axis=0)
            for k, v in prob_dict.items()
            if k[1] != PADDING_VALUE
        }

        sorted_dict = {key: prob_dict[key] for key in sorted(prob_dict.keys())}
        probabilities = np.vstack(sorted_dict.values())
        return probabilities

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.net is not None:
            prediction = self.net.predict(x_test).flatten()
            y_test = y_test.flatten()
            f1 = f1_score(y_test, prediction, average="weighted")
            return f1

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the CNN TSAnnotator to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        self.net.save(os.path.join(model_dir_path,
                      (PREDICTOR_FILE_NAME + ".keras")))
        self.net = None
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TSAnnotator":
        """Load the CNN TSAnnotator from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TSAnnotator: A new instance of the loaded CNN TSAnnotator.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        model.net = keras.saving.load_model(os.path.join(
            model_dir_path, (PREDICTOR_FILE_NAME + ".keras")))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TSAnnotationSchema,
    hyperparameters: dict,
) -> TSAnnotator:
    """
    Instantiate and train the TSAnnotator model.

    Args:
        train_data (np.ndarray): The train split from training data.
        hyperparameters (dict): Hyperparameters for the TSAnnotator.

    Returns:
        'TSAnnotator': The TSAnnotator model
    """
    model = TSAnnotator(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TSAnnotator, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TSAnnotator, predictor_dir_path: str) -> None:
    """
    Save the TSAnnotator model to disk.

    Args:
        model (TSAnnotator): The TSAnnotator model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TSAnnotator:
    """
    Load the TSAnnotator model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TSAnnotator: A new instance of the loaded TSAnnotator model.
    """
    return TSAnnotator.load(predictor_dir_path)


def evaluate_predictor_model(model: TSAnnotator, test_split: np.ndarray) -> float:
    """
    Evaluate the TSAnnotator model and return the r-squared value.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The r-squared value of the TSAnnotator model.
    """
    return model.evaluate(test_split)

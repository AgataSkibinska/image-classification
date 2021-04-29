import numpy as np
import pickle as pkl
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def check_and_swap_for_single_example(X_train: np.array,
                                      X_test: np.array,
                                      y_train: np.array,
                                      y_test: np.array,
                                      verbose: bool = False) -> Tuple[np.array, np.array, np.array, np.array]:
    single_label = 30
    if np.any(y_train == single_label):
        if verbose:
            print("No need for swap")
        pass
    else:
        if verbose:
            print("Swapping needed")
        idx = np.where(y_test == single_label)[0]
        row_to_append = X_test[idx]
        label_to_append = y_test[idx]
        X_train = np.vstack([X_train, row_to_append])
        y_train = np.vstack([y_train, label_to_append])
        X_test = np.delete(X_test, idx, axis=0)
        y_test = np.delete(y_test, idx, axis=0)

    return X_train, X_test, y_train, y_test


def prepare_data(pickle_dataset_path: str) -> Tuple[np.array, np.array, np.array, np.array]:
    X, y = pd.read_pickle(pickle_dataset_path)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = check_and_swap_for_single_example(X_train, X_test, y_train, y_test)
    return  X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true: np.array,
                          y_pred: np.array,
                          model_name: str,
                          output_path: str = None) -> None:
    c_m = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 10))
    fig = sns.heatmap(c_m, annot=True)
    fig.set_title(f'Confusion matrix for {model_name}')
    if output_path:
        fig.figure.savefig(output_path)

def print_classification_report(y_true: np.array,
                                y_pred: np.array) -> None:
    print(metrics.classification_report(y_true, y_pred))


def get_indices_wrong_predictions(y_true: np.array,
                                  y_pred: np.array,
                                  true_label: str) -> np.array:
    idxs_wrong = np.not_equal(y_true, y_pred).nonzero()[0]
    idxs_label = np.where(y_true == true_label)[0]
    idxs = np.intersect1d(idxs_wrong, idxs_label)
    return idxs

def plot_wrongly_predicted_from_class(X_test: np.array,
                                      y_true: np.array,
                                      y_pred: np.array,
                                      true_label: str,
                                      n_samples: int,
                                      shuffle: bool = False) -> None:
    idxs = get_indices_wrong_predictions(y_true, y_pred, true_label)

    if shuffle:
        np.random.shuffle(idxs)
    idxs_filtered = idxs[:n_samples]

    X_filtered = X_test[idxs_filtered][:n_samples]
    # y_true_filtered = y_true[idxs_filtered][:n_samples]
    y_pred_filtered = y_pred[idxs_filtered][:n_samples]

    if len(X_filtered) < n_samples:
        n_samples = len(X_filtered)

    X_reshaped = np.reshape(X_filtered, (-1, 56, 56))

    fig, axes = plt.subplots(1, n_samples)
    fig.suptitle(f'Examples from true class {true_label}')

    for i in range(n_samples):
        if n_samples == 1:
            axes.imshow(X_reshaped[i])
        else:
            axes[i].imshow(X_reshaped[i])
            axes[i].title.set_text(f'Predicted class {y_pred_filtered[i]}')


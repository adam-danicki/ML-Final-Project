import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.utils
from sklearn.datasets import load_digits

###---------------------------------------------------------------------
### csv processor and loader
###---------------------------------------------------------------------
#       Parameters:
#           filepath (str): Path to the dataset CSV file.
#           test_size (float)
#           train_size (float)
#       Returns:
#           train_attribs (np.ndarray): Normalized training attributes
#           test_attribs (np.ndarray): Normalized testing attributes
#           train_labels (np.ndarray): Training labels
#           test_labels (np.ndarray): Testing labels
#           cat_indices (list): Indices of categorical features
def load_and_process_csv(filepath, test_size, train_size):
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded file: {filepath}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        exit()

    labels = data['label'].copy().to_numpy()
    attribs = data.drop(columns= ['label'])

    # Categorical attribs
    for attr in attribs.columns:
        if 'cat' in attr:
            attribs[attr] = pd.factorize(attribs[attr])[0]

    cat_indices = [attribs.columns.get_loc(col) for col in attribs.columns if 'cat' in col]

    # Shuffle + split
    attribs_np, labels_np = sklearn.utils.shuffle(attribs.to_numpy(), labels)

    train_attribs, test_attribs, train_labels, test_labels = sklearn.model_selection.train_test_split(
        attribs_np, labels_np, test_size=test_size, train_size=train_size
    )

    # Normalize
    min_attr = train_attribs.min(axis=0)
    max_attr = train_attribs.max(axis=0)
    range_attr = np.where((max_attr - min_attr) == 0, 1, max_attr - min_attr)  # Prevent division by zero

    train_attribs = (train_attribs - min_attr) / range_attr
    test_attribs = (test_attribs - min_attr) / range_attr

    return train_attribs, test_attribs, train_labels, test_labels, cat_indices


###---------------------------------------------------------------------
### digits processor and loader
###---------------------------------------------------------------------
#       Parameters:
#           test_size (float)
#           train_size (float)
#       Returns:
#           train_attribs (np.ndarray): Normalized training attributes
#           test_attribs (np.ndarray): Normalized testing attributes
#           train_labels (np.ndarray): Training labels
#           test_labels (np.ndarray): Testing labels
def load_digits_dataset(test_size, train_size):
    digits = load_digits()
    attribs = digits.data
    labels = digits.target

    # Shuffle + split
    attribs, labels = shuffle(attribs, labels, random_state=42)

    train_attribs, test_attribs, train_labels, test_labels = train_test_split(
        attribs, labels, test_size=test_size, train_size=train_size
    )

    # Normalize (pixel values: 0–16)
    train_attribs = train_attribs / 16.0
    test_attribs = test_attribs / 16.0

    return train_attribs, test_attribs, train_labels, test_labels



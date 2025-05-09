import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.utils
from sklearn.datasets import load_digits

###---------------------------------------------------------------------
### csv processor and loader
###---------------------------------------------------------------------
#       Params:
#           filepath: Path to the dataset CSV file.
#           test_size
#           train_size
#       Return:
#           train_attribs: Training attributes
#           test_attribs: Testing attributes
#           train_labels: Training labels
#           test_labels: Testing labels
#           cat_indices: Indices of categorical features
def lap_csv(filepath, test_size, train_size):
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded file: {filepath}")
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}")
        exit()

    if 'label' in data.columns:
        label_col = 'label'
    elif 'Diagnosis' in data.columns:
        label_col = 'Diagnosis'

    labels_raw = data[label_col].copy().to_numpy()
    labels, _ = pd.factorize(labels_raw)
    attribs = data.drop(columns=[label_col])    

    for attr in attribs.columns:
        if 'cat' in attr:
            attribs[attr] = pd.factorize(attribs[attr])[0]

    cat_indices = [attribs.columns.get_loc(col) for col in attribs.columns if 'cat' in col]

    attribs_np, labels_np = sklearn.utils.shuffle(attribs.to_numpy(), labels)

    train_attribs, test_attribs, train_labels, test_labels = sklearn.model_selection.train_test_split(
        attribs_np, labels_np, test_size=test_size, train_size=train_size
    )

    return train_attribs, test_attribs, train_labels, test_labels, cat_indices


###---------------------------------------------------------------------
### digits processor and loader
###---------------------------------------------------------------------
#       Params:
#           test_size
#           train_size
#       Return:
#           train_attribs: Training attributes
#           test_attribs: Testing attributes
#           train_labels: Training labels
#           test_labels: Testing labels
def lap_digits(test_size, train_size):
    digits = load_digits()
    attribs = digits.data
    labels = digits.target

    attribs, labels = sklearn.utils.shuffle(attribs, labels, random_state= 42)

    train_attribs, test_attribs, train_labels, test_labels = sklearn.model_selection.train_test_split(
        attribs, labels, test_size= test_size, train_size= train_size
    )

    return train_attribs, test_attribs, train_labels, test_labels


###---------------------------------------------------------------------
### loader
###---------------------------------------------------------------------
def load_data(source, test_size, train_size):
    if source.endswith('.csv'):
        train_attribs, test_attribs, train_labels, test_labels, cat_indices = lap_csv(source, test_size, train_size)
    
    elif 'digits' in source:
        train_attribs, test_attribs, train_labels, test_labels = lap_digits(test_size, train_size)
        cat_indices = []

    return train_attribs, test_attribs, train_labels, test_labels, cat_indices
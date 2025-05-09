import numpy as np
import matplotlib.pyplot as plt
from load_and_process import load_data



### -------------------------------------------
### Dataset loading
### -------------------------------------------
attribs, _, labels, _, _ = load_data('./Data/parkinsons.csv', test_size= 0.2, train_size= 0.8)


### -------------------------------------------
### Euclidean distance
### -------------------------------------------
def euclidean_distance(feat_1, feat_2):
    return np.linalg.norm(feat_1 - feat_2)


### -------------------------------------------
### Accuracy
### -------------------------------------------
def comp_accuracy(true_labels, pred_labels):
    return np.mean(pred_labels == true_labels)


### -------------------------------------------
### Precision
### -------------------------------------------
def comp_precision(true_labels, pred_labels):
    classes = np.unique(true_labels)
    precisions = []

    for cls in classes:
        tp = np.sum((pred_labels == cls) & (true_labels == cls))
        fp = np.sum((pred_labels == cls) & (true_labels != cls))
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        precisions.append(precision)

    return np.mean(precisions)


### -------------------------------------------
### Recall
### -------------------------------------------
def comp_recall(true_labels, pred_labels):
    classes = np.unique(true_labels)
    recalls = []

    for cls in classes:
        tp = np.sum((pred_labels == cls) & (true_labels == cls))
        fn = np.sum((pred_labels != cls) & (true_labels == cls))
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        recalls.append(recall)

    return np.mean(recalls)


### -------------------------------------------
### F1
### -------------------------------------------
def comp_f1(true_labels, pred_labels):
    precision = comp_precision(true_labels, pred_labels)
    recall = comp_recall(true_labels, pred_labels)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


### -------------------------------------------
### kNN algorithm
### -------------------------------------------
def knn_algorithm(train_feats, train_labels, test_feats, k):
    predictions = []

    distances = np.array([
        [euclidean_distance(test, train) for train in train_feats]
         for test in test_feats
    ])
    
    near_feats = np.argsort(distances, axis= 1)[:, :k]
    near_labels = train_labels[near_feats]

    for entry in near_labels:
        predictions.append(np.bincount(entry).argmax())

    return np.array(predictions)


### -------------------------------------------
### Stratified Cross Validaion
### -------------------------------------------
def strat_kfold(attribs, labels, k_val):
    k = 10
    attribs = np.array(attribs)
    labels = np.array(labels)
    folds = [[] for _ in range(k)]

    for cls in np.unique(labels):
        cls_index = np.where(labels == cls)[0]
        np.random.shuffle(cls_index)
        split_parts = np.array_split(cls_index, k)
        
        for i in range(k):
            folds[i].extend(split_parts[i])
    
    folds = [np.array(fold) for fold in folds]

    accs, f1s = [], []
    for i in range(k):
        test_index = np.array(folds[i])
        train_index = np.array([idx for j in range(k) if j != i for idx in folds[j]])

        train_attribs = attribs[train_index]
        train_labels = labels[train_index]
        test_attribs = attribs[test_index]
        test_labels = labels[test_index]

        preds = knn_algorithm(train_attribs, train_labels, test_attribs, k_val)

        acc = comp_accuracy(test_labels, preds)
        f1 = comp_f1(test_labels, preds)
        accs.append(acc)
        f1s.append(f1)

    return np.mean(accs), np.mean(f1s)


### -------------------------------------------
### Experiements
### -------------------------------------------

k_vals = [1, 3, 5, 7, 9, 11, 13]
accs, f1s = [], []

print("k\tAccuracy\tF1 Score")
for k in k_vals:
    acc, f1 = strat_kfold(attribs, labels, k)
    accs.append(acc)
    f1s.append(f1)
    print(f"{k}\t{acc:.4f}\t\t{f1:.4f}")

plt.figure()
plt.plot(k_vals, accs, marker= 'o', label= 'Accuracy')
plt.plot(k_vals, f1s, marker= 's', label= 'F1 Score')
plt.title('k-NN: Accuracy and F1 Score vs k')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

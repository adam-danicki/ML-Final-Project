import numpy as np
import matplotlib.pyplot as plt
from load_and_process import load_data
from collections import Counter


### -------------------------------------------
### Dataset loading
### -------------------------------------------
train_attribs, test_attribs, train_labels, test_labels, cat_indices = load_data('./Data/digits', test_size= 0.2, train_size= 0.8)


### -------------------------------------------
### Entropy of dataset calc
### -------------------------------------------
def entropy(labels):
    _, counts = np.unique(labels, return_counts= True)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs))


### -------------------------------------------
### Info Gain calc
### -------------------------------------------
def info_gain(attribs, labels, attr_index, thresh):
    left_labels = labels[attribs[:, attr_index] <= thresh]
    right_labels = labels[attribs[:, attr_index] > thresh]
    parent_entropy = entropy(labels)

    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)
    weighted_entropy = (
        len(left_labels) / len(labels) * left_entropy +
        len(right_labels) / len(labels) * right_entropy
    )

    return parent_entropy - weighted_entropy


### -------------------------------------------
### Best split calc
### -------------------------------------------
def best_split(attribs, labels, attr_indexes):
    best_gain = 0
    best_feat = None
    best_thresh = None
    is_cat = False

    for attr_index in attr_indexes:
        vals = np.unique(attribs[:, attr_index])
        # print(f'Checking attribute index: {attr_index} with values: {vals}')

        ## If categorical
        if attr_index in cat_indices:
            #  print('Treating attribs as categorical')
             for val in vals:
                left = labels[attribs[:, attr_index] == val]
                right = labels[attribs[:, attr_index] != val]

                if len(left) == 0 or len(right) == 0:
                    continue

                gain = entropy(labels) - (
                    len(left) / len(labels) * entropy(left) +
                    len(right) / len(labels) * entropy(right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = attr_index
                    best_thresh = val
                    is_cat = True
        
        ## If numerical
        else:
            # print('Treating attribs as numerical')
            sorted_indexes = attribs[:, attr_index].argsort()
            sorted_vals = attribs[sorted_indexes, attr_index]
            sorted_labels = np.array(labels)[sorted_indexes]

            for i in range(1, len(sorted_vals)):
                if sorted_vals[i] != sorted_vals[i-1]:
                    thresh = (sorted_vals[i] + sorted_vals[i-1]) / 2
                    gain = info_gain(attribs, labels, attr_index, thresh)

                    if gain > best_gain:
                        best_gain = gain
                        best_feat = attr_index
                        best_thresh = thresh
                        is_cat = False

    # print(f"Best feature: {best_feat}\nThreshold: {best_thresh}\nIs Categorical: {is_cat}\nGain: {best_gain}")
    return best_feat, best_thresh, is_cat


### -------------------------------------------
### Build tree
### -------------------------------------------
def build_tree(attribs, labels, min_size, attr_indexes= None):
    if len(labels) == 1:
        return Counter(labels).most_common(1)[0][0]

    if len(labels) < min_size:
        return Counter(labels).most_common(1)[0][0]

    if attr_indexes is not None:
        attrs_to_consider = attr_indexes
    else:
        attrs_to_consider = range(attribs.shape[1])

    attr, thresh, is_cat = best_split(attribs, labels, attrs_to_consider)
    if attr is None:
        return Counter(labels).most_common(1)[0][0]

    if is_cat:
        left_index = attribs[:, attr] == thresh
        right_index = attribs[:, attr] != thresh
    else:
        left_index = attribs[:, attr] <= thresh
        right_index = attribs[:, attr] > thresh

    left = build_tree(attribs[left_index], labels[left_index], min_size, attr_indexes)
    right = build_tree(attribs[right_index], labels[right_index], min_size, attr_indexes)

    return {'attribute': attr, 'threshold': thresh, 'is_categorical': is_cat, 'left': left, 'right': right}


### -------------------------------------------
### Instance lable prediction
### -------------------------------------------
def predict(tree, inst):
    if not isinstance(tree, dict):
        return tree
    
    attr = tree['attribute']
    thresh = tree['threshold']
    is_cat = tree['is_categorical']

    if is_cat:
        if inst[attr] == thresh:
            return predict(tree['left'], inst)
        else:
            return predict(tree['right'], inst)
    else:
        if inst[attr] <= thresh:
            return predict(tree['left'], inst)
        else:
            return predict(tree['right'], inst)
        
### -------------------------------------------
### Build tree and predictor test
### -------------------------------------------
# simple_tree = build_tree(train_attribs, train_labels)

# test_instance = test_attribs[0]
# true_label = test_labels.iloc[0] if isinstance(test_labels, pd.Series) else test_labels[0]

# predicted_label = predict(simple_tree, test_instance)

# print(f"Test Instance: {test_instance}")
# print(f"True Label: {true_label}")
# print(f"Predicted Label: {predicted_label}")


### -------------------------------------------
### Bootstrap method
### -------------------------------------------
def bootstrap(attribs, labels):
    indexes = np.random.choice(len(attribs), size= len(attribs), replace= True)
    return attribs[indexes], labels[indexes]

### -------------------------------------------
### Majority vote
### -------------------------------------------
def major_vote(predictions):
    return Counter(predictions).most_common(1)[0][0]


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
### Random Forest
### -------------------------------------------
def rand_forest(train_attribs, train_labels, test_attribs, ntree, min_size):
    m = int(np.sqrt(train_attribs.shape[1]))
    forest = []

    for _ in range(ntree):
        samp_attribs, samp_labels = bootstrap(train_attribs, train_labels)
        attr_subset = np.random.choice(train_attribs.shape[1], m, replace=False)
        tree = build_tree(samp_attribs, samp_labels, min_size=min_size, attr_indexes=attr_subset)
        forest.append((tree, attr_subset))

    final_predicts = []
    for inst in test_attribs:
        tree_preds = [predict(tree, inst) for tree, _ in forest]
        final_predicts.append(major_vote(tree_preds))

    return np.array(final_predicts)


### -------------------------------------------
### Stratified Cross Validaion
### -------------------------------------------
def strat_kfold(attribs, labels, ntree, min_size):
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
        test_idex = np.array(folds[i])
        train_idex = np.array([idex for j in range(k) if j != i for idex in folds[j]])

        trainf_attribs_raw, trainf_labels = attribs[train_idex], labels[train_idex]
        testf_attribs_raw, testf_labels = attribs[test_idex], labels[test_idex]

        min_attr = trainf_attribs_raw.min(axis= 0)
        max_attr = trainf_attribs_raw.max(axis= 0)
        range_attr = np.where((max_attr - min_attr) == 0, 1, max_attr - min_attr)

        trainf_attribs = (trainf_attribs_raw - min_attr) / range_attr
        testf_attribs = (testf_attribs_raw - min_attr) / range_attr

        preds = rand_forest(trainf_attribs, trainf_labels, testf_attribs, ntree, min_size= min_size)

        accs.append(comp_accuracy(testf_labels, preds))
        f1s.append(comp_f1(testf_labels, preds))

    return np.mean(accs), np.mean(f1s)


### -------------------------------------------
### Experiments
### -------------------------------------------
min_size = 3
ntree_vals = [1, 5, 10, 20, 30, 40]
accs, f1s = [], []

print("ntree\tAccuracy\tF1 Score")
for ntree in ntree_vals:
    acc, f1 = strat_kfold(train_attribs, train_labels, ntree, min_size)
    accs.append(acc)
    f1s.append(f1)
    print(f"{ntree}\t{acc:.4f}\t\t{f1:.4f}")

plt.figure()
plt.plot(ntree_vals, accs, marker= 'o', label= 'Accuracy')
plt.plot(ntree_vals, f1s, marker= 's', label= 'F1 Score')
plt.title('Random Forest: Accuracy and F1 Score vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Score')
plt.grid(True)
plt.legend()
plt.show()

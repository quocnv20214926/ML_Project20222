import numpy as np

def calculate_entropy(y):
    classes = np.unique(y)
    entropy=0
    for cls in classes:
        p_cls= len(y[y==cls])/len(y)
        entropy +=-p_cls*np.log2(p_cls)
    return entropy
def calculate_left_value(y):
    classes, counts = np.unique(y,return_counts =True)
    idx = np.argmax(counts)
    return classes[idx]
def calculate_information_gain(current_entropy, left_y, right_y):
    

    left_entropy = calculate_entropy(left_y)
    right_entropy = calculate_entropy(right_y)


    n = len(left_y) + len(right_y)


    info_gain = current_entropy - (len(left_y) / n * left_entropy + len(right_y) / n * right_entropy)

    return info_gain
def find_best_split(X, y, num_features, current_entropy, min_samples_split):
    
    best_feature = None
    best_value = None
    max_info_gain = -float("inf")
    for feature in range(num_features):
        feature_values = X[:, feature]

        unique_values = np.unique(feature_values)


        for value in unique_values:
            left_indices = np.where(feature_values <= value)[0]
            right_indices = np.where(feature_values > value)[0]
            if len(left_indices) < min_samples_split or len(right_indices) < min_samples_split:
                continue
            left_y = y[left_indices]
            right_y = y[right_indices]
            info_gain = calculate_information_gain(current_entropy, left_y, right_y)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
                best_value = value
                left_X, left_y, right_X, right_y = X[left_indices], y[left_indices], X[right_indices], y[right_indices]

    return best_feature, best_value, left_X, left_y, right_X, right_y
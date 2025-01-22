# import pickle

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

def filter_classes_with_few_samples(data, labels):
    """
    Filters out classes that have fewer than 2 samples.
    """
    label_counts = Counter(labels)
    filtered_data = []
    filtered_labels = []

    for i in range(len(data)):
        if label_counts[labels[i]] > 1:  # Include only classes with more than 1 sample
            filtered_data.append(data[i])
            filtered_labels.append(labels[i])

    return np.array(filtered_data), np.array(filtered_labels)

# Load single-hand data
single_hand_data_dict = pickle.load(open('./single_hand_data.pickle', 'rb'))
single_hand_data = np.asarray(single_hand_data_dict['data'])
single_hand_labels = np.asarray(single_hand_data_dict['labels'])

# Load double-hand data
double_hand_data_dict = pickle.load(open('./double_hand_data.pickle', 'rb'))
double_hand_data = np.asarray(double_hand_data_dict['data'])
double_hand_labels = np.asarray(double_hand_data_dict['labels'])

# Filter single-hand data to remove classes with fewer than 2 samples
single_hand_data, single_hand_labels = filter_classes_with_few_samples(single_hand_data, single_hand_labels)

# Filter double-hand data to remove classes with fewer than 2 samples
double_hand_data, double_hand_labels = filter_classes_with_few_samples(double_hand_data, double_hand_labels)

# Train model for single-hand gestures
if len(single_hand_data) > 0:
    # Split the data into training and testing sets
    x_train_single, x_test_single, y_train_single, y_test_single = train_test_split(
        single_hand_data, single_hand_labels, test_size=0.2, shuffle=True, stratify=single_hand_labels
    )

    # Train RandomForest Classifier for single hand
    model_single_hand = RandomForestClassifier()
    model_single_hand.fit(x_train_single, y_train_single)

    # Evaluate model
    y_predict_single = model_single_hand.predict(x_test_single)
    score_single = accuracy_score(y_predict_single, y_test_single)
    print('{}% of single-hand samples were classified correctly!'.format(score_single * 100))

    # Save the trained model for single-hand detection
    with open('model_single_hand.p', 'wb') as f:
        pickle.dump({'model': model_single_hand}, f)
else:
    print("No data available for single-hand detection after filtering. Skipping training for single hand.")

# Train model for double-hand gestures
if len(double_hand_data) > 0:
    # Split the data into training and testing sets
    x_train_double, x_test_double, y_train_double, y_test_double = train_test_split(
        double_hand_data, double_hand_labels, test_size=0.2, shuffle=True, stratify=double_hand_labels
    )

    # Train RandomForest Classifier for double hand
    model_double_hand = RandomForestClassifier()
    model_double_hand.fit(x_train_double, y_train_double)

    # Evaluate model
    y_predict_double = model_double_hand.predict(x_test_double)
    score_double = accuracy_score(y_predict_double, y_test_double)
    print('{}% of double-hand samples were classified correctly!'.format(score_double * 100))

    # Save the trained model for double-hand detection
    with open('model_double_hand.p', 'wb') as f:
        pickle.dump({'model': model_double_hand}, f)
else:
    print("No data available for double-hand detection after filtering. Skipping training for double hand.")

#Creating some features based on the bytes, and then using those to train the classifier

import requests
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

url = "https://raw.githubusercontent.com/daniel-h-0/comp_detect_csv/master/dataset_0.csv"

response = requests.get(url, stream=True)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text))

#create new dataset from byte data. the new dataset will consist of features interpreted from each row of bytes, such as entropy, mean, etc
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

def calculate_entropy(byte_sequence):
    value_counts = np.bincount(byte_sequence, minlength=256)
    probabilities = value_counts / len(byte_sequence)
    return sum(p * np.log2(p) for p in probabilities if p > 0)

def create_feature_row(row):
    byte_array = np.array(row, dtype=np.uint8)
    entropy = calculate_entropy(byte_array)
    mean = byte_array.mean()
    std_dev = byte_array.std()
    unique_count = len(np.unique(byte_array))
    most_common = np.bincount(byte_array).max()
    value_range = byte_array.max() - byte_array.min()
    
    return [entropy, mean, std_dev, unique_count, most_common, value_range]

def create_features(input):
    new_dataset = []
    for row in input:
        feature_row = create_feature_row(row)
        new_dataset.append(feature_row)
    
    return new_dataset
features = X.apply(create_feature_row, axis=1)
#features = create_features(X)

print(features.shape)
feature_matrix = np.array(features.tolist())

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=0.2)

model = MLPClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


import requests
import pandas as pd
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
print(df.shape)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
print(X.shape)
print(y.shape)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

model = MLPClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Unnormalised:
#               precision    recall  f1-score   support

#            0       0.49      1.00      0.66      1645
#            1       0.50      0.00      0.01      1714

#     accuracy                           0.49      3359
#    macro avg       0.49      0.50      0.33      3359
# weighted avg       0.49      0.49      0.33      3359

# Normalised:
#               precision    recall  f1-score   support

#            0       0.56      0.58      0.57      1682
#            1       0.56      0.54      0.55      1677

#     accuracy                           0.56      3359
#    macro avg       0.56      0.56      0.56      3359
# weighted avg       0.56      0.56      0.56      3359
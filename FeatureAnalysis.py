#Creating some features based on the bytes, and then using those to train the classifier

import requests
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import scipy.stats

url = "https://raw.githubusercontent.com/daniel-h-0/comp_detect_csv/master/dataset_0.csv"

response = requests.get(url, stream=True)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text))


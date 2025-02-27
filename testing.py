import requests
import pandas as pd
from io import StringIO

url = "https://raw.githubusercontent.com/daniel-h-0/comp_detect_csv/master/dataset_0.csv"

response = requests.get(url, stream=True)
response.raise_for_status()

df = pd.read_csv(StringIO(response.text))

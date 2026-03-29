import json
from collections import Counter

data = json.load(open("data/training_data.json"))
print(len(data))
labels = [row["label"] for row in data]
print(Counter(labels))
print(data[0])

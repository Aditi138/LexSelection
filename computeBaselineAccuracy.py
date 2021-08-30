import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="en-es")
parser.add_argument("--word", type=str, default="wall_NOUN")
args = parser.parse_args()

if __name__ == "__main__":
    test_file = args.input + "/" + args.word + "/" + args.word + ".new.test.features"
    test_data = pd.read_csv(test_file, sep=',')
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(test_data[["label"]])
    label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    id2label = {v: k for k, v in label2id.items()}
    print(label2id)

    test_label = label_encoder.transform(test_data[["label"]])
    counter = Counter(test_label)

    total_samples = len(test_label)
    max_baseline = 0.0
    acc = {}
    label = None
    for class_ in range(len(counter)):
        value = counter[class_]
        acc = value * 1.0 / total_samples
        if acc > max_baseline:
            max_baseline = acc
            label= id2label[class_]

    print(max_baseline,label )


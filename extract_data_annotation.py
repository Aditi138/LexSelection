import pandas as pd
import numpy as np
np.random.seed(1)
import argparse
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="en-es/")
parser.add_argument("--word", type=str, default="wall")
parser.add_argument("--pos", type=str,default="NOUN")
args = parser.parse_args()

def printOneExample(orig, tgt, source_word, tgt_words):
    orig = orig.split()
    orig[source_word] = "<b>" + orig[source_word] + "</b>"
    tgt = tgt.split()
    if tgt_words == "nan":
        orig, tgt = " ".join(orig), " ".join(tgt)
        return orig, tgt

    for tgt_index in tgt_words.split(","):
        tgt_index = int(float(tgt_index))
        tgt[tgt_index] = "<b>" + tgt[tgt_index] + "</b>"
    orig, tgt = " ".join(orig), " ".join(tgt)
    return orig, tgt

if __name__ == "__main__":
    input_dir = args.input + f'/{args.word}/'
    train_input = input_dir + f'{args.word}.train.features'
    test_input = input_dir + f'{args.word}.test.features'
    required_columns_file = input_dir + f'{args.word}.col.names'
    args.pos = args.word.split("_")[1]

    print(f"Reading existing train/test from {train_input}")
    all_train, all_test = pd.read_csv(train_input, sep=","), pd.read_csv(test_input, sep=",")
    with open(required_columns_file, 'r') as fin:
        lines = fin.readlines()
        new_columns = lines[0].strip().split("\t")
        label_columns = lines[1].strip().split("\t")
    print(f"Train {all_train.shape}, Test {all_test.shape}, Classes {len(label_columns)}.")

    # Get labels
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(all_train[["label"]])
    label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    id2label = {v: k for k, v in label2id.items()}
    print(label2id)

    all_test_features, all_test_label = all_test[new_columns], label_encoder.transform(all_test[["label"]])
    label_list = []
    for i in range(len(label2id.keys())):
        label_list.append(id2label[i])

    #For each label create data points and shuffle them
    data = []
    uniq = set()
    data_templata = {'id': None, 'src_word': args.word, 'pos':args.pos, 'cur_src': [], 'tgt': [], 'target_words': label_list,
                     'correct': None, 'expl': []}
    common  =0 

    for label, label_id in label2id.items():
        sent = 0
        required_indices = np.where(all_test_label == label_id)[0]
        for index in required_indices:
            source, target, src_word_index, tgt_word_indices = all_test.iloc[index]["orig_sentence"], all_test.iloc[index]["tgt_translation"], \
                                                               all_test.iloc[index]["source_word"], all_test.iloc[index]["tgt_word"]
            source,target = printOneExample(source, target, src_word_index, str(tgt_word_indices))
            
            if target in uniq:
                common += 1
                continue
            datapoint = deepcopy(data_templata)
            datapoint['id'] = len(data)
            datapoint['src_word'] = args.word
            datapoint['cur_src'].append(source)
            datapoint['tgt'].append(target)
            datapoint['correct'] = int(label_id)
            data.append(datapoint)
            sent += 1
            uniq.add(target)

        print(f"Added {sent} data points for {label_id} : {label}")

    print(len(uniq), len(data), common)
    random.shuffle(data)
    jsonoutput = input_dir + f"/{args.word}_anno.json"
    print(f"Data for annotation in: {jsonoutput} ")
    with open(jsonoutput, 'w') as fout:
        json.dump(data, fout)

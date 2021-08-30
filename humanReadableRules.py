import codecs
import argparse
from collections import defaultdict
import numpy as np
from nltk.corpus import wordnet as wn
import json
from copy import deepcopy
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,default='en-es/')
parser.add_argument("--feature_map", type=str, default="/feature_map")
parser.add_argument("--word", type=str, default="farmer_NOUN")
parser.add_argument("--human_selected", type=str, default=None)
parser.add_argument("--max", type=int, default=40)
args = parser.parse_args()

random.seed(1001)

def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def printExamples(examples_per_features, fout, lines_per_label, feature_names):
    num = 5
    examples = set()
    for feat_name in feature_names:
        examples_lines = examples_per_features[feat_name]
        for eg in examples_lines:
            info = lines_per_label[eg].strip().split("\t")
            label_id, label_, features, orig, tgt, source_word, tgt_words = int(info[0]), info[1], info[2].split(
                ","), info[3], info[4], int(info[5]), info[6]
            orig = orig.split()
            orig[source_word] = "<label>" + orig[source_word] + "</label>"
            tgt = tgt.split()
            if tgt_words == "nan":
                continue
            for tgt_index in tgt_words.split(","):
                tgt_index = int(float(tgt_index))
                tgt[tgt_index] = "<label>" + tgt[tgt_index] + "</label>"
            orig, tgt = " ".join(orig), " ".join(tgt)
            examples.add((orig, tgt))

    examples = list(examples)
    index_list = [i for i in range(len(examples))]
    num = min(num, len(index_list))
    index = np.random.choice(index_list, num, replace=False)
    for i in index:
        (orig, tgt) = examples[i]
        fout.write("SRC: " + orig + " -->\tTGT: " + tgt + "\n")

def outputData(examples_per_features, top_feat_names, lines_per_label, feature_names, cur_label_id, text_sumary):
    explantions = []
    for top_feature in top_feat_names:
        explantions.append(text_sumary[top_feature])

    covered = set()
    all_examples_sorted_by_top_features = []
    for feature in top_feat_names:
        examples = set()
        for feat_name in feature_names[feature]:
            examples_lines = examples_per_features[feat_name]
            for eg in examples_lines:
                if eg in covered:
                    continue
                covered.add(eg)
                info = lines_per_label[eg].strip().split("\t")
                label_id, label_, features, orig, tgt, source_word, tgt_words = int(info[0]), info[1], info[2].split(
                    ","), info[3], info[4], int(info[5]), info[6]
                orig = orig.split()
                orig[source_word] = "**" + orig[source_word] + "**"
                tgt = tgt.split()
                if tgt_words == "nan":
                    continue
                for tgt_index in tgt_words.split(","):
                    tgt_index = int(float(tgt_index))
                    tgt[tgt_index] = "**" + tgt[tgt_index] + "**"
                orig, tgt = " ".join(orig), " ".join(tgt)
                examples.add((orig, tgt))
        all_examples_sorted_by_top_features += list(examples)


    return all_examples_sorted_by_top_features, explantions

def createDescriptionWithExamples(top_features, top_feature_effects, top_feature_values, fout , examples_per_features, lines_per_label, label_id):
    feature_summary = defaultdict(list)
    feature_names = defaultdict(list)
    top_feat_names = []
    for feat_name, feat_effect, feat_value in zip(top_features, top_feature_effects, top_feature_values):
        print(feat_name, np.round(feat_effect,3), feat_value)
        feature_info = feat_name.split("__")
        feature, feature_name = feature_info[0].split(".")[0], feature_info[-1] #ignore the position of the index in the vicinity
        feature_summary[feature].append((feature_name, feat_value))
        feature_names[feature].append(feat_name)
        top_feat_names.append(feature)

    text_summary = {}
    for feature, summary in feature_summary.items():
        feature_info = feature_map[feature]
        pos, neg = set(),set()
        text = ""

        for (name, value) in summary:

            if "wsd" in feature:
                token = name.split(".")[0]
                name = token + ": " + wn.synset(name)._definition
            if value > 0.0:
                pos.add(name)
            else:
                neg.add(name)
        fout.write(feature_info + " --> \t")
        text += feature_info + " "
        if len(pos) > 0:
            pos= list(pos)
            fout.write("IN: " + ", ".join(pos) + "\t   ")
            text += "IN: " + ", ".join(pos) + " "

        if len(neg) > 0:
            if len(pos) > 0:
                text += "BUT "
            neg = list(neg)
            fout.write("NOT IN: " +   ", ".join(neg) + "\t")
            text += "NOT IN: " + ", ".join(neg)

        text += "\n"
        fout.write("\n")
        text_summary[feature] = text
        printExamples(examples_per_features, fout, lines_per_label, feature_names[feature])
        fout.write("\n")

    fout.write("\n")
    print()
    all_examples_per_features_sorted, explanations = outputData(examples_per_features, top_feat_names, lines_per_label, feature_names, label_id, text_summary)
    return all_examples_per_features_sorted, explanations

def getExamples(top_features, lines_per_label, top_feature_values):
    examples_per_features = defaultdict(set)
    for line_num, line in enumerate(lines_per_label):
        info = line.strip().split("\t")
        label_id, label_, features, orig, tgt, source_word, tgt_words = int(info[0]), info[1], info[2].split(","), info[3], info[4], int(info[5]), info[6].split(",")
        printed_feat = []
        i = 0
        while i < len(features):
            feat_name, feat_effect, feat_value, i = parse(features, i)
            if feat_name in top_features:

                index = top_features.index(feat_name)
                feature_value = top_feature_values[index]
                if feat_value == feature_value:
                    printed_feat.append(feat_name + "= " + str(feat_effect))
                    examples_per_features[feat_name].add(line_num)


    return examples_per_features

def readImportantFeatures(debug_file, examples_per_label_feature, output_file, id2label):
    explanations = {}
    explanation_map = {}
    individual_feature_explansion = {}
    with open(debug_file, 'r') as fin, open(output_file, 'w') as fout:
        print(f"Reading features from {debug_file} and outputting readable rules to {output_file}")
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split(";")
            label_info = info[0].split(",")
            _, label = int(label_info[0]), label_info[1]
            feature_names = info[-1].split("~~~")
            if label not in label2id:
                continue
            label_id = label2id[label]


            explanation_map[label_id] = {}
            fout.write(label + "\n")
            expl = {}
            covered = set()
            features_combined = defaultdict(list)
            original_features = defaultdict(list)
            for feat_name  in feature_names: #Ordered list
                if feat_name == "":
                    continue
                feature_info = feat_name.split("__")
                feature, feature_name = feature_info[0].split(".")[0], feature_info[-1]  # ignore the position of the index in the vicinity
                if "head_lemma" in feature or "dep_lemma" in feature:
                    feature = 'dep_lemma'
                elif "head_wsd" in feature or "dep_wsd" in feature:
                    feature = 'dep_wsd'

                key = feature_map[feature]
                value = feature_name
                if "wsd" in feature:
                    value_name = wn.synset(value)._definition
                    value = f'\'{value.split(".")[0]}\'' + " as in " + value_name



                if (key,value) not in covered:
                    features_combined[key].append(value)
                    original_features[key].append(feat_name)
                    fout.write(key + " -> " + value + "\n")


                    examples = examples_per_label_feature[label_id][feat_name]
                    random.shuffle(examples)
                    for (feat_indexs, orig, tgt, src_word, tgt_words) in examples[:3]:
                        orig, tgt = printOneExample(orig, tgt, src_word, tgt_words)
                        fout.write(orig + " --> " + tgt + "\n")
                    fout.write("\n")

                covered.add((key,value))
                # if len(expl) >= 5: #only top 5  features
                #     break

            fout.write("\n")

            for feature_key, values in features_combined.items():
                expl[feature_key] = (feature_key, values, original_features[feature_key])
            explanations[label_id] = expl

    return explanations, individual_feature_explansion

def readExamples(input_file):
    label2id = {}
    examples_per_label_id = defaultdict(list)
    examples_per_label_feature = {}

    covered = set()
    with open(input_file, 'r') as fin:
        print(f'Reading examples from {input_file}')
        lines = fin.readlines()
        for line in lines:
            info = line.strip().split("\t")
            _, label, features, feature_index,  orig_sent, tgt_sent, src_word, tgt_word = int(info[0]), info[1], info[2].split("~~~"), info[3].split("~~~"), info[4], info[5], int(info[6]),info[7]

            if label not in human_selected:
                continue
            else:
                if label not in label2id:
                    label2id[label] = len(label2id)

                label_id = label2id[label]
                human_selected_examples = human_selected[label]
                found = False
                for (src, tgt) in human_selected_examples:
                    if src == orig_sent and tgt == tgt_sent:
                        found= True
                        break

                if found and (orig_sent not in covered) and (len(examples_per_label_id[label_id]) < args.max):

                    examples_per_label_id[label_id].append((features, feature_index, orig_sent, tgt_sent, src_word, tgt_word))
                    if label_id not in examples_per_label_feature:
                        examples_per_label_feature[label_id] = defaultdict(list)
                    for feature in features:
                        examples_per_label_feature[label_id][feature].append((feature_index, orig_sent, tgt_sent, src_word, tgt_word))
                    covered.add(orig_sent)


    for label_id, eg in examples_per_label_id.items():
        print(label_id, len(eg))

    return examples_per_label_id, examples_per_label_feature, label2id

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

def readHumanSelected(input):
    if not input:
        return {}
    with open(input, 'r') as fin:
        word_examples = defaultdict(list)
        for line in fin.readlines():
            info = line.strip().split("@@@")
            word, src, tgt, correct, num_agree = info[0], info[1], info[2], info[3], info[-1]
            if int(num_agree) == 4 and word == args.word: #all 3 annotators agreeing with the correct
                src = src.replace("<b>","").replace("</b>","")
                tgt = tgt.replace("<b>","").replace("</b>","")
                word_examples[correct].append((src, tgt))

    for label, examples in word_examples.items():
        print(f"Examples for label: {label} = {len(examples)}")
        random.shuffle(examples)

    new_word_examples = {}
    for label in word_examples.keys():
        if label in ['píldora']:
            continue
        examples = word_examples[label]
        new_word_examples[label] = examples
        print(f"Examples for label after threshold: {label} = {len(new_word_examples[label])}")

    return new_word_examples

if __name__ == "__main__":
    input_dir = args.input + "/" + args.word
    input_file = input_dir + "/" + 'important.datapoints.test.txt'
    debug_file = input_dir  + "/" + 'important.features'
    output_file = input_dir + "/" + 'examples.txt'
    with open(args.feature_map, 'r') as fin:
        lines = fin.readlines()
        feature_map = {}
        for line in lines:
            line = line.strip().split(" = ")
            feature_map[line[0]] = line[-1]

    human_selected = readHumanSelected(args.human_selected)


    examples_per_label_id, examples_per_label_feature , label2id = readExamples(input_file) #Read individual example sentences for each label


    id2label = {v:k for k,v in label2id.items()}
    print(id2label)
    explanations, individual_feature_explansion = readImportantFeatures(debug_file, examples_per_label_feature, output_file, id2label) #outputting examples for each feature/label


    label_list = []

    for i in range(len(label2id.keys())):
        label_list.append(id2label[i])

    #Creating data for annotation project
    data = []
    data_templata = {'id' : None, 'src_word': args.word, 'cur_src': [], 'tgt': [], 'target_words': label_list, 'correct': None, 'expl': []}
    examples_covered_label = defaultdict(list)

    id = 0
    for label_id, examples in examples_per_label_id.items():
        random.shuffle(examples)
        for example in examples:
            (features,features_index, orig, tgt, src_word, tgt_words) = example
            orig, tgt = printOneExample(orig, tgt, src_word, tgt_words)
            datapoint = deepcopy(data_templata)
            datapoint['id'] = int(id)
            datapoint['cur_src'].append(orig)
            datapoint['tgt'].append(tgt)
            datapoint['correct'] = int(label_id)
            datapoint['index'] = []

            expl = []
            feature_expanded = []
            bolded_per_examples = []
            feature_value_all =[]
            feature_index_set = []

            for feature, (key, value, original_features) in explanations[label_id].items():
                bolded_per_example = []
                feature_index_per_example = []
                expl.append(key  + " --> " + "~~~".join(value))



                for j,f in enumerate(features):
                    for index,o in enumerate(original_features):
                        if f == o: #If feature present in the rule:

                            bolded_per_example.append(str(index))
                            feature_index_per_example.append(features_index[j])
                            break


                if len(bolded_per_example) == 0: #No features in the expl present in the example
                    bolded_per_example = "NA"
                    feature_index_per_example = "NA"
                else:
                    bolded_per_example = "~~~".join(bolded_per_example)
                    feature_index_per_example = "~~~".join(feature_index_per_example)

                bolded_per_examples.append(bolded_per_example)
                feature_expanded.append(key)
                feature_value_all.append("~~~".join(value))
                feature_index_set.append(feature_index_per_example)


            datapoint['expl'] = "\n".join(expl)
            datapoint['bolded'] = bolded_per_examples
            datapoint['feature_expanded'] = feature_expanded
            datapoint['feature_value_all_list'] = feature_value_all
            datapoint['index'] = feature_index_set
            data.append(datapoint)

            id += 1


    random.shuffle(data)
    new_datapoints = []
    index_remaining = []
    features_per_class = defaultdict(lambda : 0)
    for index, d in enumerate(data):
        non_empty=False
        for b in d['bolded']:
            if b !='NA':
                non_empty=True
        if non_empty:
            new_datapoints.append(d)
            features_per_class[id2label[d['correct']]] += 1
        else:
            index_remaining.append(index)

    print(f"Examples with features: {len(new_datapoints)}")
    print(features_per_class)

    for index in index_remaining:
        new_datapoints.append(data[index])


    jsonoutput = input_dir + "/examples.json"
    print(f"Data for annotation in: {jsonoutput} ")
    with open(jsonoutput, 'w') as fout:
        json.dump(data, fout)






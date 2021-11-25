import codecs
from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
import tqdm
import pandas as pd
import iml
import re
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt

en_stops = set(stopwords.words('english'))
punctuations = ['!', '"', '#', '$', '%', '&', '\\', '(', ')', '*', '+',
                ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
                '\\', ']', '^', '_', '`', '{', '|', '}', '~', '»']
to_remove_list = list(en_stops) + punctuations + ['be']


class FeatureLoader(object):
    def __init__(self):
        self.vocab2id = {}

    def addorGetId(self, w):
        if pd.isna(w):
            if 'UNK' in self.vocab2id:
                return self.vocab2id['UNK']

            self.vocab2id['UNK'] = len(self.vocab2id)
            return self.vocab2id['UNK']
        w = w.lower()
        if w in self.vocab2id:
            return self.vocab2id[w]
        else:
            self.vocab2id[w] = len(self.vocab2id)
            return self.vocab2id[w]

def gen_line(filename):
    if not filename:
        return None
    with open(filename) as f:
        for line in f:
            yield line.strip()

def stopWord(token):
    if token in to_remove_list:
        return True
    return False

def cleanText(token):

    new_token = ""
    for c in token:
        if c not in punctuations:
            new_token += c
    return new_token

def removeStopWordsPunctuations(token):
    token = cleanText(token)
    if token in to_remove_list:
        return True
    return False

def parseAnalysis(info):
    info = info.strip().split(" ||| ")
    if len(info) == 2: #greek has lemma ||| text
        lemma, text = info[0].lower().split(), info[1].lower().split()
        return lemma, None, None, None, None, text
    lemma, head, deprel, feat, upos, text = info[0].lower().split(), info[1].split(), info[3].split(), info[2].split(), info[4].split(), info[5].lower().split()
    new_lemma = []
    if len(lemma) != len(text):
        return None, None, None,None, None,None
    for num, lem in enumerate(lemma):
        if '-pron-' in lem:
            new_lemma.append(text[num])
        else:
            new_lemma.append(lem)

    return new_lemma, head, deprel, feat, upos, text

def parseAnalysisTarget(info, original, sent_num):
    info = info.lower().strip().split(" ||| ") #this input is from target spacy
    lemma, orig = info[0].split(), info[1].replace(" |||", "").split()

    if len(lemma) == len(original): #The tokenization of lemma and the original is same so return lemma, spacy tokenization could be different 
        return lemma

    aligned_lemma = [""] * len(original)
    index = 0
    for token_num, token in enumerate(original):
        tokenized_orig = orig[token_num]
        found = False
        try:
            while not found:
                if index >= len(orig):
                    # print(sent_num, original, lemma_tokens, wsd)
                    break

                if tokenized_orig in token or token in tokenized_orig:
                    aligned_lemma[index] = lemma[token_num]
                    found = True
                index += 1
        except:
            print(f' WSD Error in {sent_num}')

    for token_num, token in enumerate(aligned_lemma):
        if token == "":
            aligned_lemma[token_num] = original[token_num]

    return aligned_lemma


def parseAnalysisSource(lemma, orig, original, sent_num, poses):
    if len(lemma) == len(original): #The tokenization of lemma and the original is same so return lemma, spacy tokenization could be different
        return lemma, poses

    aligned_lemma = [""] * len(original)
    index = 0
    prev_index = -1
    orig_index = 0
    aligned_pos = [""] * len(original)

    for token_num, tokenized_orig in enumerate(orig): #Iterating the tokenized text
        if orig_index >= len(original):
            break
        token = original[orig_index]
        if tokenized_orig in token or token in tokenized_orig:
            aligned_lemma[index] = lemma[token_num]
            aligned_pos[index] = poses[token_num]
            found = True
            prev_index = index
            orig_index += 1
            index += 1


    for token_num, token in enumerate(aligned_lemma):
        if token == "":
            aligned_lemma[token_num] = original[token_num]
            aligned_pos[token_num] = poses[token_num]

    return aligned_lemma, aligned_pos

def readAnalysis(info):
    info = info.strip().split(" ||| ")
    lemma, head, deprel, feat, upos = info[0].lower().split(), info[1].split(), info[3].split(), info[2].split(), \
                                            info[4].split()
    return lemma, head, deprel, feat, upos

def parseAlignment(info):
    src_tgt_alignments = defaultdict(list)
    tgt_src_alignments = defaultdict(list)
    for align in info:
        s = int(align.split('-')[0])
        t = int(align.split('-')[1])
        src_tgt_alignments[s].append(t)
        tgt_src_alignments[t].append(s)
    return src_tgt_alignments, tgt_src_alignments

def readAlignments(input):
    src, tgt = [], []
    with codecs.open(input, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            info = line.strip().replace("p","-").split()
            src_tgt_alignments, tgt_src_alignments = parseAlignment(info)
            src.append(src_tgt_alignments)
            tgt.append(tgt_src_alignments)
    return src, tgt

def alignWSD(original, lemma_tokens, wsd, sent_num):
    aligned_wsd = [""] * len(original)
    aligned_wsd_index = 0
    for token_num, token in enumerate(original):
        #token = cleanText(token)
        lemma_token = lemma_tokens[token_num]
        found = False
        try:
            while not found:
                if aligned_wsd_index >= len(wsd):
                    #print(sent_num, original, lemma_tokens, wsd)
                    break
                info = wsd[aligned_wsd_index].split("@#*")
                orig, lemma= info[0].lower(), info[1].lower()
                if token in orig or lemma_token in lemma or lemma in lemma_token or orig in token:
                    aligned_wsd[token_num] = wsd[aligned_wsd_index]
                    found = True
                aligned_wsd_index += 1
        except:
            print(f' WSD Error in {sent_num}')

    for token_num, token in enumerate(aligned_wsd):
        if token == "":
            aligned_wsd[token_num] = wsd[token_num]
    return aligned_wsd

def isContiguous(indices):
    not_contiuous = False
    source = indices[0]
    for index in indices[1:]:
        if index - source > 1:
            not_contiuous = True
            break
        source = index
    return not_contiuous


def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return preds, labels

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

def calculate_top_contributors(shap_values, features=None, feature_names=None, use_abs=False, return_df=False,
                               n_features=5):
    """ Adapted from the SHAP package for visualizing the contributions of features towards a prediction.
        https://github.com/slundberg/shap
        Args:
            shap_values: np.array of floats
            features: pandas.core.series.Series, the data with the values
            feature_names: list, all the feature names/ column names
            use_abs: bool, if True, will sort the data by the absolute value of the feature effect
            return_df: bool, if True, will return a pandas dataframe, else will return a list of feature, effect, value
            n_features: int, the number of features to report on. If it equals -1 it will return the entire dataframe
        Returns:
            if return_df is True: returns a pandas dataframe
            if return_df is False: returns a flattened list by name, effect, and value
        """
    assert not type(shap_values) == list, "The shap_values arg looks looks multi output, try shap_values[i]"
    assert len(shap_values.shape) == 1, "Expected just one row. Please only submit one row at a time."

    shap_values = np.reshape(shap_values, (1, len(shap_values)))
    instance = iml.Instance(np.zeros((1, len(feature_names))), features)
    link = iml.links.convert_to_link('identity')

    # explanation obj
    expl = iml.explanations.AdditiveExplanation(
        shap_values[0, -1],                 # base value
        np.sum(shap_values[0, :]),          # this row's prediction value
        shap_values[0, :-1],                # matrix
        None,
        instance,                           # <iml.common.Instance object >
        link,                               # 'identity'
        iml.Model(None, ["output value"]),  # <iml.common.Model object >
        iml.datatypes.DenseData(np.zeros((1, len(feature_names))), list(feature_names))
    )

    # Get the name, effect and value for each feature, if there was an effect
    features_ = {}
    for i in range(len(expl.data.group_names)):
        if expl.effects[i] != 0:
            value = ensure_not_numpy(expl.instance.group_display_values[i])
            if value > 0.0: #Get top active features
                features_[i] = {
                    "effect": ensure_not_numpy(expl.effects[i]),
                    "value": value,
                    "name": expl.data.group_names[i]
                }
    if len(features_) == 0:
        return None

    effect_df = pd.DataFrame([v for k, v in features_.items()])

    if use_abs:  # get the absolute value of effect
        effect_df['abs_effect'] = effect_df['effect'].apply(np.abs)
        effect_df.sort_values('abs_effect', ascending=False, inplace=True)
    else:
        effect_df.sort_values('effect', ascending=False, inplace=True)
    if not n_features == -1:
        effect_df = effect_df.head(n_features)
    if return_df:
        return effect_df.reset_index(drop=True)
    else:
        list_of_info = list(zip(effect_df.name, effect_df.effect, effect_df.value))
        effect_list = list(sum(list_of_info, ()))  # flattens the list of tuples
        return effect_list



def ensure_not_numpy(x):
    """Helper function borrowed from the iml package"""
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str):
        return str(x)
    elif isinstance(x, np.generic):
        return float(np.asscalar(x))
    else:
        return x

def extractLemma(token, wsd_token):
    info = wsd_token.split("@#*")
    orig, lemma = info[0].lower(),info[1]
    if '-PRON-' in lemma:
        return orig
    return lemma.lower()

def extractSense(token, wsd_token, lemma):
    if "wn:" in wsd_token:
        word_net = wsd_token.split("wn:")[-1]
        syn = wn.of2ss(word_net)._name
    else:
        syn = 'NA'
    return  syn

def alignTgtLemma(orig_target, tgt_lemmas, original_target, source_upos=None):
    if not source_upos:
        source_upos = None
    if len(orig_target) == len(original_target):
        return orig_target, tgt_lemmas, source_upos
    targets, lemmas, poses = ["" for _ in range(len(original_target))], ["" for _ in range(len(original_target))], ["" for _ in range(len(original_target))]

    num = 0

    special=False
    for orig_token_num, orig_token in enumerate(original_target):#Check if the original target had those special characters
        if len(re.findall(r'[\u0080-\u0099]',orig_token)) == 1 and len(orig_token) == 1:
            special=True
            break
    if not special:
        return orig_target, tgt_lemmas, source_upos

    for orig_token_num, orig_token in enumerate(original_target): #Iterate on the original target which might contain the erroneous input
        if len(re.findall(r'[\u0080-\u0099]',orig_token)) == 1 and len(orig_token) == 1: #found the erroneous character
            continue
        else:
            target, lemma = orig_target[num], tgt_lemmas[num]
            targets[orig_token_num] = target
            lemmas[orig_token_num] = lemma
            if source_upos:
                poses[orig_token_num] = source_upos[num]
            num += 1
    return targets, lemmas, poses

def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()

def plot_coefficients_label(classifier, feature_names, label_list, word, top_features=20):
    coef = classifier.coef_
    num_classes = coef.shape[0]
    important_features = {}
    if num_classes > 2:
        for class_ in range(num_classes):
            label = label_list[class_].split("/")[0]
            print(label)
            coefficients = coef[class_,:]
            top_positive_coefficients = np.argsort(coefficients)[-top_features:]
            top_negative_coefficients = np.argsort(coefficients)[:top_features]
            top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
            # create plot
            plt.figure(figsize=(15, 5))
            colors = ['red' if c < 0 else 'blue' for c in coefficients[top_coefficients]]
            plt.bar(np.arange(2 * top_features), coefficients[top_coefficients], color=colors)

            feature_names = np.array(feature_names)
            required_features = feature_names[top_coefficients][-top_features:]
            print("\n".join([str(r) for r in required_features]))
            plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
            #plt.show()
            print()

            plt.savefig(f'./{label}.pdf')
            important_features[class_] = required_features
    else:
        coef = coef.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        #plt.show()
        #plt.savefig(f'./{word}.pdf')

        #Class = 0
        label = label_list[0].split("/")[0]
        print(label)
        required_features = list(feature_names[top_negative_coefficients])
        required_features.reverse()
        print("\n".join([str(r) for r in required_features]) + "\n")
        important_features[0] = required_features

        # Class = 1
        label = label_list[1].split("/")[0]
        print(label)
        required_features = list(feature_names[top_positive_coefficients])
        required_features.reverse()
        print("\n".join([str(r) for r in required_features]))
        important_features[1] = required_features

    return important_features


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    imp, names = imp[:20], names[:20]
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


def filter_words(source_pos_target_tokens, target_source_tokens, target_freq_threshold, fileprefix,combine_lemmas=False):
    updated_source_pos_target_tokens = {}

    unambig = 0
    source_pos_tokens = defaultdict(lambda :0)

    for (lemma, pos), tgt_words in source_pos_target_tokens.items():

        normalized_tokens = []
        tgt_tokens = []
        freq_tokens = defaultdict(lambda: 0)
        mwt = False
        for tgt, count in tgt_words.items():
            if count < target_freq_threshold:  # remove tgt translations which occur less than 5 times (alignment errors)
                continue

            if checkMultipleAlignments(tgt, target_source_tokens):
                continue
            mwt = mwt or len(tgt.split()) > 1

            # if "..." in tgt or len(
            #         tgt.split()) > 1 or tgt in lemma:  # Skip for multi-word translations and have same translation as source
            #     continue
            if "..." in tgt or tgt in lemma or lemma in tgt:  # Skip for multi-word translations and have same translation as source
                continue
            freq_tokens[tgt] = count
            tgt_tokens.append(tgt)
            normalized_tokens.append(unidecode.unidecode(tgt))


        if len(tgt_tokens) < 1 and not mwt: #If a word has 1:1 mapping we skip it
            unambig += 1
            continue


        if len(tgt_tokens) == 0:
            continue
        prev_tokens = [[tgt_tokens[0]]]
        class_index = 0
        for token_num in range(1, len(tgt_tokens)):
            cur_token = tgt_tokens[token_num]
            same_class = False
            for prev_token in prev_tokens[class_index]:
                paths = [prev_token, cur_token]
                un_prev_token, un_cur_token = unidecode.unidecode(prev_token), unidecode.unidecode(cur_token)
                paths = [un_prev_token, un_cur_token]
                prefix = os.path.commonprefix(paths)
                ed = editdistance.eval(un_prev_token, un_cur_token)
                t = int(0.6 * len(un_cur_token))
                if len(
                        prefix) >= t or un_cur_token in un_prev_token or un_prev_token in un_cur_token:  # combine these classes together\
                    same_class = True
                    break

            if combine_lemmas and same_class:
                prev_tokens[class_index].append(cur_token)
            else:
                class_index += 1
                prev_tokens.append([cur_token])

        if len(prev_tokens) > 0:  # After compression there are mutliple tokens:
            output = []
            total_count = 0
            for tgt_tokens in prev_tokens:
                representation = "/".join(tgt_tokens)
                count = 0
                for t in tgt_tokens:
                    count += freq_tokens[t]
                    total_count += freq_tokens[t]
                if count > 50:  # A target token should have more than 50 sentences
                    output.append((representation, count))

            if len(output) > 0:
                updated_source_pos_target_tokens[(lemma, pos)] = {}
                for (tgt_token, count) in output:
                    updated_source_pos_target_tokens[(lemma, pos)][tgt_token] = count
                    source_pos_tokens[(lemma, pos)] += count

    print(f'Before: {len(source_pos_target_tokens)} After: {len(updated_source_pos_target_tokens)} Unambiguous tokens : {unambig}')

    #Retained after filtering
    with open(f"./allparallelwords_{fileprefix}.debug", 'w') as fout:
        for (source,pos), tgt_tokens in updated_source_pos_target_tokens.items():
            fout.write(source + "," + pos + "," + str(source_pos_tokens[(source,pos)]) + "-->\t")
            for tgt_token, count in tgt_tokens.items():
                fout.write(tgt_token + "= " + str(count) + "; ")
            fout.write("\n")
    return updated_source_pos_target_tokens, source_pos_tokens
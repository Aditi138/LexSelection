import argparse
from collections import defaultdict
import codecs
from nltk.corpus import wordnet as wn
import pandas as pd
import os
from utils import *
import spacy
import  numpy as np

nlp = spacy.load('en_core_web_sm')
lemmatizer =  nlp.vocab.morphology.lemmatizer

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="/en-es/")
parser.add_argument("--word", type=str, default='language_NOUN')
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--remove_wsd", action="store_true", default=False, help="To remove WSD features set to True")

args = parser.parse_args()

def createVocab(cols, vocab):
    for col in cols:
        for w in data[col].unique().tolist():
            wid = vocab.addorGetId(w)

def inspectFile():
    input = "/en-es/language_NOUN/language_NOUN.new.features"
    data = pd.read_csv(input, sep=',')
    example = data.iloc[0]

    values = example.values > 0
    orig = example['orig_sentence']
    tgt = example['tgt_translation']
    label = example['label']
    source_word = example['source_word']
    sent_num = example['sent_num']


def getTrainTest(train_data, test_data):
    train_index = []
    test_index = []

    train_index  = train_data['sent_num'].unique().tolist()
    test_index = test_data['sent_num'].unique().tolist()

    inter = set(train_index) & set(test_index)
    train_index =  list(set(train_index) - inter)
    inter = set(train_index) & set(test_index)
    assert len(inter) == 0
    #train_index = [int(i) for i in train_index]
    #test_index = [int(i) for i in test_index]
    return train_index, test_index

def split_train_test():
    input_dir =args.input  + "/" + args.word + "/"
    train_input = input_dir + f'{args.word}.train.features'
    test_input = input_dir + f'{args.word}.test.features'

    if not os.path.exists(train_input):
        data_one_hot, new_columns, label_columns = convertNumericalToCategorical(data)


        all_train, all_test = [], []
        for target_word in label_columns:
            all_features = data_one_hot[data_one_hot[target_word] == 1]
            # remove 20% from the all_features as test
            train = all_features.sample(frac=0.8, random_state=args.seed)  # random state is a seed value
            test = all_features.drop(train.index)
            all_train.append(train)
            all_test.append(test)

        all_train = pd.concat(all_train)
        all_test = pd.concat(all_test)
        all_train.to_csv(train_input)
        all_test.to_csv(test_input)

    print(f"Train {all_train.shape}, Test {all_test.shape}.")
    return all_train, all_test

def convertNumericalToCategorical(df):
    print("Starting DF shape: %d, %d" % df.shape)
    columns = ["head_pos","pos","deprel","head_lemma","wsd","head_wsd",
               "del_rel","dep_pos","dep_lemma","dep_wsd",
               "dep_lemma.1","dep_wsd.1","dep_lemma.2","dep_wsd.2","dep_lemma.3","dep_wsd.3",
               "lemma-bigram","lemma-bigram.1","lemma-bigram.2","lemma-bigram.3","label"]
    label_columns = []
    new_columns = ["is_dep", "is_dep.1"]
    for col in columns:
        s = df[col].unique()
        # Create a One Hot Dataframe with 1 row for each unique value
        one_hot_df = pd.get_dummies(s, prefix='%s_' % col)
        one_hot_df[col] = s

        #print("Adding One Hot values for %s (the column has %d unique values)" % (col, len(s)))
        pre_len = len(df)

        # Merge the one hot columns
        df = df.merge(one_hot_df, on=[col], how="left")
        assert len(df) == pre_len
        #print(df.shape)
        if col == 'label':
            label_columns = list(one_hot_df.columns[:-1])
        else:
            new_columns += list(one_hot_df.columns[:-1])
    #print(new_columns, label_columns, df.shape)
    return df, new_columns, label_columns
if __name__ == "__main__":
    # inspectFile()
    # exit(-1)

    input_file = args.input + "/" + args.word  + "/" + args.word + ".features"
    input_file_token_nums = args.input + "/" + args.word + "/" + args.word + ".features.token.nums"
    train_file = args.input + "/" + args.word  + "/" + args.word + ".train.features"
    test_file = args.input + "/" + args.word  + "/" + args.word + ".test.features"

    train_output_file =  args.input + "/" + args.word  + "/" + args.word + ".new.train.features"
    train_output_file_tokens = args.input + "/" + args.word + "/" + args.word + ".new.train.features.token.nums"

    test_output_file = args.input + "/" + args.word + "/" + args.word + ".new.test.features"
    test_output_file_tokens = args.input + "/" + args.word + "/" + args.word + ".new.test.features.token.nums"

    data = pd.read_csv(input_file, sep=",")
    data_token_nums = pd.read_csv(input_file_token_nums, sep=',')
    
    os.system(f'rm -rf {train_output_file} {train_output_file_tokens} {test_output_file} {test_output_file_tokens}')
    if not os.path.exists(train_file):
        print("Creating train/test split")
        
        split_train_test()

    train_data = pd.read_csv(train_file, sep=',')
    test_data = pd.read_csv(test_file, sep=',')
    print('Read train/test split: ', train_file, test_file)
    train_index, test_index = getTrainTest(train_data, test_data)
    print(f"Read train/test split: {len(train_index)}, {len(test_index)}")


    #Create Vocabulary
    #Pos vocabulary
    POSVocab = FeatureLoader()
    createVocab(cols = ['head_pos', 'pos', 'dep_pos', 'dep_pos.1', 'dep_pos.2', 'dep_pos.3', 'dep_pos.4', 'dep_pos.5'], vocab=POSVocab)

    RelVocab = FeatureLoader()
    relcols = ['deprel','del_rel', 'del_rel.1', 'del_rel.2', 'del_rel.3', 'del_rel.4', 'del_rel.5']
    createVocab(cols = relcols, vocab=RelVocab)

    LemmaVocab = FeatureLoader()
    lemmacols = ['head_lemma', 'dep_lemma', 'dep_lemma.1', 'dep_lemma.2', 'dep_lemma.3', 'dep_lemma.4', 'dep_lemma.5']
    createVocab(cols=lemmacols, vocab=LemmaVocab)

    if not args.remove_wsd:
        WSDvocab = FeatureLoader()
        wsdcols = ['head_wsd', 'dep_wsd', 'dep_wsd.1', 'dep_wsd.2', 'dep_wsd.3', 'dep_wsd.4', 'dep_wsd.5']
        createVocab(cols = wsdcols, vocab=WSDvocab)

    LemmaBigramVocab = FeatureLoader()
    lemmabigramcols = ['lemma-bigram', 'lemma-bigram.1', 'lemma-bigram.2', 'lemma-bigram.3']
    createVocab(cols = lemmabigramcols, vocab=LemmaBigramVocab)

    # print(POSVocab.vocab2id)
    # print(LemmaVocab.vocab2id)
    # print(WSDvocab.vocab2id)
    # print(LemmaBigramVocab.vocab2id)
    # print(RelVocab.vocab2id)


    pos_id2vocab = {v:k for k,v in POSVocab.vocab2id.items()}
    lemma_id2vocab = {v: k for k, v in LemmaVocab.vocab2id.items()}
    if not args.remove_wsd:
        wsd_id2vocab = {v: k for k, v in WSDvocab.vocab2id.items()}
    lemmabigram_id2vocab = {v: k for k, v in LemmaBigramVocab.vocab2id.items()}
    rel_id2vocab = {v: k for k, v in RelVocab.vocab2id.items()}

    head_pos_features = [] #POS of the head of the source word
    prefix = 'head_pos__'
    for i in range(len(POSVocab.vocab2id.keys())):
        head_pos_features.append(f'{prefix}{pos_id2vocab[i]}')

    rel_features = [] #Dependency relation with head pos
    prefix = 'deprel__'
    for i in range(len(RelVocab.vocab2id.keys())):
        rel_features.append(f'{prefix}{rel_id2vocab[i]}')

    nearby_lemma_features = []
    prefix = 'lemma__'
    for i in range(len(LemmaVocab.vocab2id.keys())):
        nearby_lemma_features.append(f'{prefix}{lemma_id2vocab[i]}')

    nearby_wsd_features = []
    if not args.remove_wsd:
        prefix = 'wsd__'
        for i in range(len(WSDvocab.vocab2id.keys())):
            nearby_wsd_features.append(f'{prefix}{wsd_id2vocab[i]}')

    nearby_bigram_features  = []
    prefix = 'bigram__'
    for i in range(len(LemmaBigramVocab.vocab2id.keys())):
        nearby_bigram_features.append(f'{prefix}{lemmabigram_id2vocab[i]}')


    columns = head_pos_features + rel_features + nearby_lemma_features + nearby_bigram_features + nearby_wsd_features
    columns += ['orig_sentence', 'tgt_translation',  'label', 'source_word', 'tgt_word', 'sent_num']
    print(f"Number of features : {len(columns)}")

    covered = set()




    for index in range(len(data)):
        d = data.iloc[index]
        d_token = data_token_nums.iloc[index]
        sent_num = d['sent_num']
        assert sent_num == d_token['sent_num']
        if sent_num in covered:
            continue
        covered.add(sent_num)

        head_pos_features_numpy = [0] * len(head_pos_features)
        rel_features_numpy = [0] * len(rel_features)
        nearby_lemma_features_numpy =[0] * len(nearby_lemma_features)
        nearby_bigram_features_numpy = [0] * len(nearby_bigram_features)
        nearby_wsd_features_numpy = [0] * len(nearby_wsd_features)

        head_pos_features_tokens = [-1] * len(head_pos_features)
        rel_features_tokens = [-1] * len(rel_features)
        nearby_lemma_features_tokens = [-1] * len(nearby_lemma_features)
        nearby_bigram_features_tokens = [-1] * len(nearby_bigram_features)
        nearby_wsd_features_tokens = [-1] * len(nearby_wsd_features)

        head_pos_id = POSVocab.addorGetId(d['head_pos'])
        head_pos_features_numpy[head_pos_id] = 1
        token_id = d_token['head_pos']
        head_pos_features_tokens[head_pos_id] = token_id

        rel_id = RelVocab.addorGetId(d['deprel'])
        rel_features_numpy[rel_id] = 1
        token_id = d_token['deprel']
        rel_features_tokens[rel_id]= token_id

        for col in lemmacols:
            lemma_id = LemmaVocab.addorGetId(d[col])
            token_id = d_token[col]
            nearby_lemma_features_tokens[lemma_id]  = token_id
            nearby_lemma_features_numpy[lemma_id] = 1 #If any of the lemmas in the vicnity is active

        for col in lemmabigramcols:
            bigram_id = LemmaBigramVocab.addorGetId(d[col])
            nearby_bigram_features_numpy[bigram_id] = 1  # If any of the lemma-bigram in the vicnity is active
            token_id = d_token[col]
            nearby_bigram_features_tokens[bigram_id] = token_id

        if not args.remove_wsd:
            for col in wsdcols:
                wsd_id = WSDvocab.addorGetId(d[col])
                nearby_wsd_features_numpy[wsd_id] = 1  # If any of the wsd in the vicnity is active
                token_id = d_token[col]
                nearby_wsd_features_tokens[wsd_id] =  token_id


        all_features = head_pos_features_numpy + rel_features_numpy + nearby_lemma_features_numpy + nearby_bigram_features_numpy + nearby_wsd_features_numpy

        orig, tgt, label, source_word, tgt_word, sent_num= d['orig_sentence'], d['tgt_translation'], d['label'], d['source_word'], d['tgt_word'], d['sent_num']
        all_features.append(orig)
        all_features.append(tgt)
        all_features.append(label)
        all_features.append(source_word)
        all_features.append(tgt_word)
        all_features.append(sent_num)

        token_features = head_pos_features_tokens + rel_features_tokens + nearby_lemma_features_tokens + nearby_bigram_features_tokens + nearby_wsd_features_tokens
        token_features.append(orig)
        token_features.append(tgt)
        token_features.append(label)
        token_features.append(source_word)
        token_features.append(tgt_word)
        token_features.append(sent_num)

        if sent_num in train_index:
            df = pd.DataFrame([all_features], columns=columns)
            df.to_csv(train_output_file, mode='a', header=not os.path.exists(train_output_file))

            dfindex = pd.DataFrame([token_features], columns=columns)
            dfindex.to_csv(train_output_file_tokens, mode='a', header=not os.path.exists(train_output_file_tokens))
        else:
            dft = pd.DataFrame([all_features], columns=columns)
            dft.to_csv(test_output_file, mode='a', header=not os.path.exists(test_output_file))

            dfindext = pd.DataFrame([token_features], columns=columns)
            dfindext.to_csv(test_output_file_tokens, mode='a', header=not os.path.exists(test_output_file_tokens))
        if index % 100 == 0:
            print(f"Processed {index}")


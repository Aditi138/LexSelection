import argparse
from collections import defaultdict
import codecs
from nltk.corpus import wordnet as wn
import pandas as pd
import os
from utils import *
import spacy

nlp = spacy.load('en_core_web_sm')
lemmatizer =  nlp.vocab.morphology.lemmatizer

parser = argparse.ArgumentParser()
parser.add_argument("--orig_input", type=str, default="eng-spa")
parser.add_argument("--input", type=str, default="eng-spa")
parser.add_argument("--input_words", type=str, default="source_target_words")
parser.add_argument("--alignments", type=str, default="eng-spa.pred")
parser.add_argument("--features", type=str, default="eng-spa.all.features")
parser.add_argument("--wsd", type=str, default="eng-spa.wsd")
parser.add_argument("--source_analysis", type=str, default="eng-spa.analysis")
parser.add_argument("--target_analysis", type=str, default="eng-spa.spa.analysis")
parser.add_argument("--prune", action="store_true", default=False)
args = parser.parse_args()

class Input(object):
    def __init__(self, source_words, pos, target_words):
        self.source_words = source_words
        self.pos = pos
        self.target_words = target_words

def gen_line(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

def readInput(input):
    with codecs.open(input, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        source_pos_target = []
        for line in lines:
            if line.startswith('#'):
                continue
            info = line.strip().split(",")
            source_words = [info[0].lstrip().rstrip()]
            pos = info[1].lstrip().rstrip()
            tgt_words = info[-1].strip().rstrip().split(";")

            input_obj = Input(source_words, pos, tgt_words)
            source_pos_target.append(input_obj)
        return source_pos_target

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

def parseAlignment(info):
    src_tgt_alignments = defaultdict(list)
    tgt_src_alignments = defaultdict(list)
    for align in info:
        s = int(align.split('-')[0])
        t = int(align.split('-')[1])
        src_tgt_alignments[s].append(t)
        tgt_src_alignments[t].append(s)
    return src_tgt_alignments, tgt_src_alignments

def getTargetAlignedTokens(sent_num, tgt_lemma, token_num, alignments_src):
    tgt_words = []
    tgt_indices = alignments_src[token_num]
    for index in tgt_indices:
        tgt_words.append(tgt_lemma[index])
    tgt_word = " ".join(tgt_words)
    tgt_indices = [str(t) for t in tgt_indices]
    tgt_indices = ",".join(tgt_indices)
    return tgt_word, tgt_indices

def extractSense(token, wsd_token, lemma):
    if "wn:" in wsd_token:
        word_net = wsd_token.split("wn:")[-1]
        syn = wn.of2ss(word_net)._name
    else:
        syn = 'NA'
    return  syn

def extractLemma(token, wsd_token):
    info = wsd_token.split("@#*")
    orig, lemma = info[0].lower(),info[1]
    if '-PRON-' in lemma:
        return orig
    return lemma

def extract_features(sent_num, token_num, align_wsd_token, original_src_sentence, upos, heads, deprel, lemma):
    wsd_sense = extractSense(token_num , align_wsd_token[token_num], lemma[token_num])
    indices = [i for i in range(len(original_src_sentence))]

    features = []
    token_nums = []

    #consider context +/- 3 tokens surrounding the ambiguous word
    left_token_index = max(0, token_num - 3)
    right_token_index = min(len(original_src_sentence)-1, token_num+3)

    #get all deps of ambiguous token
    dependent_token_indices = []
    for h,d,i in zip(heads, deprel, indices):
        h = int(h)
        if h-1 == token_num: #The current tokens' head is the source word
            dependent_token_indices.append(i)

    head_source_token = int(heads[token_num])
    head_relation = deprel[token_num]
    if head_source_token > 0:
        head_pos = upos[head_source_token-1]
        wsd_head_sense = extractSense(head_source_token-1 , align_wsd_token[head_source_token-1], lemma[head_source_token-1])
        lemma_head_pos = extractLemma(head_source_token-1, align_wsd_token[head_source_token-1])
    else:
        head_pos = 'root'
        lemma_head_pos = 'root'
        wsd_head_sense = 'root'

    features.append(head_pos) #POS of the head of source token
    features.append(upos[token_num]) #POS of the source token
    features.append(head_relation) #relation between the source token and head
    features.append(lemma_head_pos)
    features.append(wsd_sense) #WSD of the source token
    features.append(wsd_head_sense) #WSD sense of the head

    token_nums.append(head_source_token-1)
    token_nums.append(token_num)
    token_nums.append(head_source_token - 1)
    token_nums.append(head_source_token - 1)
    token_nums.append(token_num)
    token_nums.append(head_source_token - 1)

    #take either 6 dependen
    #Get pos, deprel if the ambig token is head or dep
    num_pos_filled = 6 #Total context features
    dependents = set()
    for token_index in range(left_token_index, right_token_index+1):
        if token_index == token_num:
            continue

        if token_index in dependent_token_indices:
            features.append(1) #Is dependent
            features.append(deprel[token_index]) #relation of the dependent
            dependents.add(token_index)

            token_nums.append(token_index)
            token_nums.append(token_index)

        else:
            features.append(0) #Is not dependent but in vicinity
            features.append('NA')

            token_nums.append(token_index)
            token_nums.append(token_index)

        if token_index >= len(original_src_sentence):
            print(original_src_sentence, token_index, len(upos))

        features.append(upos[token_index])  # POS of the depdendent
        token_nums.append(token_index)

        lemma_ = extractLemma(token_index, align_wsd_token[token_index])
        features.append(lemma_)  # lemma of the dependent
        token_nums.append(token_index)

        wsd_sense = extractSense(token_index, align_wsd_token[token_index],
                                 lemma[token_index])
        features.append(wsd_sense)  # wsd sense of the dep
        token_nums.append(token_index)
        num_pos_filled -= 1

    if num_pos_filled > 0:
        for token_index in dependent_token_indices:
            if token_index not in dependents and num_pos_filled > 0:
                features.append(1)  # Is dependent
                features.append(deprel[token_index])
                features.append(upos[token_index])
                lemma_ = extractLemma(token_index, align_wsd_token[token_index])
                features.append(lemma_)
                wsd_sense = extractSense(token_index, align_wsd_token[token_index],
                                         lemma[token_index])
                features.append(wsd_sense)
                num_pos_filled -= 1

                token_nums.append(token_index)
                token_nums.append(token_index)
                token_nums.append(token_index)
                token_nums.append(token_index)
                token_nums.append(token_index)

    while num_pos_filled > 0:
        features.append(0)
        features.append('NA')
        features.append('NA')
        features.append('NA')
        features.append('NA')

        token_nums.append(-1)
        token_nums.append(-1)
        token_nums.append(-1)
        token_nums.append(-1)
        token_nums.append(-1)
        num_pos_filled -= 1

    #collocation features, bi-gram features of lemma
    stop_words_removed = []
    context_position = []
    original_index = []
    for token_index in range(len(original_src_sentence)):
        if token_index == token_num:
            lemma_ = extractLemma(token_index, align_wsd_token[token_index])
            stop_words_removed.append(lemma_)
            context_position.append(0)
            original_index.append(token_index)
            continue
        lemma_token = extractLemma(token_index, align_wsd_token[token_index])
        if not stopWord(lemma_token):
            stop_words_removed.append(lemma_token)
            context_position.append(token_index - token_num)
            original_index.append(token_index)


    #consider collocation windoe size of 2, since we have removed stop words, limit = 2*2 = 4
    num_filled = 0
    token_index = context_position.index(0)
    for index in range(token_index-2, token_index+3):
        if index == token_index:
            continue
        if index >=  0 and index < len(context_position) and index+1 < len(context_position):
            set_context = (stop_words_removed[index], stop_words_removed[index+1])
            features.append(set_context)
            token_nums.append(str(original_index[index]) +"," + str(original_index[index+1]))
            num_filled += 1

    while 4-num_filled > 0:
        features.append('NA')
        token_nums.append(-1)
        num_filled += 1

    return features, token_nums

def isValid(source_words, input):
    for word in source_words:
        if word == input:
            return True
    return False

def isValidTgt(tgt_word, label2idindividual, tgt_lemma):
    if len(tgt_lemma) ==0:
        return False, -1
    for alt in label2idindividual.keys():
        if alt == tgt_word:
            return True, label2idindividual[alt]
    for tgt_token in tgt_lemma:
        for alt in label2idindividual.keys():
            if alt == tgt_token:
                return True, label2idindividual[alt]
    return False, -1

def extract_sentences(source_pos_target, original_src_sentence, original_tgt_translation, alignments_src, alignments_tgt, src_lemma, source_heads, source_deprels,uposes,
                          tgt_lemma, src_wsd_input, sent_num):

    try:
        align_wsd_token = alignWSD(original_src_sentence, src_lemma, src_wsd_input, sent_num)
        for token_num, orig_token in enumerate(original_src_sentence):
            pos = uposes[token_num]
            lemma_token = extractLemma(token_num, align_wsd_token[token_num]) #extract lemma for the english source side

            if isValid(source_words,lemma_token) and pos == source_pos:  # (lemma, pos) present in the required words
                tgt_word, tgt_indices_string = getTargetAlignedTokens(sent_num, tgt_lemma, token_num, alignments_src)
                isvalid, label = isValidTgt(tgt_word, label2idindividual, tgt_lemma
)
                if isvalid:  # Tgt word in the required words
                    #align_wsd_token = alignWSD(original_src_sentence, src_lemma, src_wsd_input, sent_num)
                    label_name = id2label[label]
                    # extract the features for this data sample from the source sentence
                    features, token_nums = extract_features(sent_num, token_num, align_wsd_token, original_src_sentence, uposes, source_heads, source_deprels, src_lemma)
                    assert len(features) == len(token_nums)

                    features.append(" ".join(original_src_sentence))
                    features.append(" ".join(original_tgt_translation))
                    features.append(label_name)
                    features.append(str(token_num))
                    features.append(tgt_indices_string)
                    features.append(str(sent_num))

                    token_nums.append(" ".join(original_src_sentence))
                    token_nums.append(" ".join(original_tgt_translation))
                    token_nums.append(label_name)
                    token_nums.append(str(token_num))
                    token_nums.append(tgt_indices_string)
                    token_nums.append(str(sent_num))

                    assert len(features) == total_features
                    assert len(token_nums) == total_features

                    df = pd.DataFrame([features], columns=columns)
                    df.to_csv(args.features, mode='a', header=not os.path.exists(args.features))

                    df_ = pd.DataFrame([token_nums], columns=columns)
                    df_.to_csv(args.features + ".token.nums", mode='a', header=not os.path.exists(args.features + ".token.nums"))
                    
    except Exception as e:
        print(f'Error in {sent_num}', e)

if __name__ == "__main__":
    input_dir = os.path.dirname(args.input)
    source_pos_target = readInput(args.input_words)

    # create training
    for input_obj  in source_pos_target:
        source_words = input_obj.source_words
        source_pos = input_obj.pos
        target_words = input_obj.target_words
        print(f"Processing {source_words[0]} {source_pos} -- {target_words}")
        word_pos = source_words[0] + "_"  + source_pos
        args.features = input_dir + "/" + word_pos + "/" + word_pos + ".features"
        print(f"Outputting features in {args.features}")
        sent_num = 0
        os.system(f"rm -rf {input_dir}/{word_pos}/*")
        os.system(f"mkdir -p {input_dir}/{word_pos}/")


        filenames =[args.input, args.alignments, args.source_analysis, args.target_analysis, args.wsd, args.orig_input]
        gens = [gen_line(n) for n in filenames]
        
        # Create the target_classes
        label2id, label2idindividual = {}, {}
        for target_word in target_words:
            target_word = target_word.lstrip().rstrip()
            label2id[target_word] = len(label2id)
            for tgt_word in target_word.split("/"):
                label2idindividual[tgt_word] = label2id[target_word]
        id2label = {v: k for k, v in label2id.items()}
        columns = ['head_pos', 'pos', 'deprel', 'head_lemma', 'wsd', 'head_wsd']
        columns += ['is_dep', 'del_rel', 'dep_pos', 'dep_lemma', 'dep_wsd'] *  6 + \
                   ['lemma-bigram']*4 + \
                   ['orig_sentence', 'tgt_translation', 'label', 'source_word', 'tgt_word', 'sent_num']

        total_features = len(columns)
        for input, alignment, source_analysis, target_analysis, wsd_file , original_input in zip(*gens):
            original = input.lower().strip().split(" ||| ") #Cleaned input
            orig_source, orig_tgt = original[0].split(), original[1].split()

            original_input = original_input.lower().strip().split(" ||| ")  # Original input may have special characters
            original_source, original_target = original_input[0].split(), original_input[1].split()

            alignments_src, alignments_tgt = parseAlignment( alignment.strip().replace("p","-").split())
            source_lemmas, source_heads, source_deprels, source_feats, source_uposes = parseAnalysis(source_analysis)
            #tgt_lemmas, tgt_heads, tgt_deprels, tgt_feats, tgt_uposes = parseAnalysis(target_analysis)
            tgt_lemmas = parseAnalysisTarget(target_analysis, orig_tgt, sent_num)
            source_wsd = wsd_file.strip().split()

            #Clean original source and skip sentences with <=5 words
            clean_tokens = []
            for token in orig_source:
                if token not in punctuations:
                    clean_tokens.append(token)
            if len(clean_tokens) < 3 and args.prune:
                sent_num +=1 
                continue

            orig_target, tgt_lemmas, _ = alignTgtLemma(orig_tgt, tgt_lemmas, original_target)
            orig_source, source_lemmas, source_uposes = alignTgtLemma(orig_source, source_lemmas, original_source,
                                                                      source_upos=source_uposes)

            if len(orig_source) != len(source_lemmas) or len(orig_target) != len(tgt_lemmas):
                print("Lemma mismatch", sent_num)
                sent_num += 1
                continue

            if len(source_lemmas) != len(source_uposes):
                print("UPOS mismatch", sent_num)
                sent_num += 1
                continue

            extract_sentences(source_pos_target, orig_source, orig_target,
                                                 alignments_src, alignments_tgt,
                                                 source_lemmas, source_heads, source_deprels, source_uposes,
                                                 tgt_lemmas,
                                                 source_wsd,
                                                 sent_num)
            sent_num += 1
            if sent_num % 100000 == 0:
                print(f"Processed {sent_num}")



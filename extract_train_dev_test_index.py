import argparse
from collections import defaultdict
import codecs
from nltk.corpus import wordnet as wn
import pandas as pd
import os
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--orig_input", type=str, default="eng-spa")
parser.add_argument("--input", type=str, default="eng-spa")
parser.add_argument("--input_words", type=str, default="source_target_words")
parser.add_argument("--alignments", type=str, default="eng-spa.pred")
parser.add_argument("--features", type=str, default="eng-spa.all.features")
parser.add_argument("--wsd", type=str, default="eng-spa.wsd")
parser.add_argument("--use_wsd", action="store_true", default=False, help="If you have WSD for L1, set it to True")
parser.add_argument("--source_analysis", type=str, default="eng-spa.analysis")
parser.add_argument("--target_analysis", type=str, default="eng-spa.spa.analysis")
parser.add_argument("--prune", action="store_true", default=False)
parser.add_argument("--stopwords", type=str, default="/stopwords")
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


def readStopWords():
    stopwords = []
    with open(args.stopwords, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


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
        # token = cleanText(token)
        lemma_token = lemma_tokens[token_num]
        found = False
        try:
            while not found:
                if aligned_wsd_index >= len(wsd):
                    # print(sent_num, original, lemma_tokens, wsd)
                    break
                info = wsd[aligned_wsd_index].split("@#*")
                orig, lemma = info[0].lower(), info[1].lower()
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
    tgt_indices_string = [str(t) for t in tgt_indices]
    tgt_indices_string = "@".join(tgt_indices_string)
    return tgt_word, tgt_indices, tgt_indices_string


def extractSense(token, wsd_token, lemma):
    if "wn:" in wsd_token:
        word_net = wsd_token.split("wn:")[-1]
        syn = wn.of2ss(word_net)._name
    else:
        syn = 'NA'
    return syn


def extractLemma(token, wsd_token):
    info = wsd_token.split("@#*")
    orig, lemma = info[0].lower(), info[1]
    if '-PRON-' in lemma:
        return orig
    return lemma


def extract_features(sent_num, token_num, tgt_indices, align_wsd_token,
                     original_src_sentence, upos, heads, deprel, lemma,
                     original_tgt_sentence, tgt_upos, tgt_heads, tgt_deprel, tgt_lemma):
    stopwords = readStopWords()

    if args.reverse:
        L1_sentence = original_tgt_sentence
        L1_upos = tgt_upos
        L1_heads, L1_deprel, L1_lemma = tgt_heads, tgt_deprel, tgt_lemma
        L1_token_num = tgt_indices[0]
        L1_indices = tgt_indices
    else:
        L1_sentence = original_src_sentence
        L1_upos = upos
        L1_heads, L1_deprel, L1_lemma = heads, deprel, lemma
        L1_token_num = token_num
        L1_indices = [L1_token_num]

    if args.use_wsd:
        wsd_sense = extractSense(L1_token_num, align_wsd_token[L1_token_num], L1_lemma[L1_token_num])

    indices = [i for i in range(len(L1_sentence))]

    features = []
    token_nums = []

    # get all deps of ambiguous token
    dependent_token_indices = []
    for h, d, i in zip(L1_heads, L1_deprel, indices):
        h = int(h)
        if h - 1 in tgt_indices:  # The current tokens' head is the source word
            dependent_token_indices.append(i)


    head_source_token = int(L1_heads[L1_token_num])
    head_relation = L1_deprel[L1_token_num]
    if head_source_token > 0:
        head_pos = L1_upos[head_source_token - 1]
        lemma_head_pos = L1_lemma[head_source_token - 1]
        orig_head_pos = L1_sentence[head_source_token - 1]
    else:
        head_pos = 'X'
        lemma_head_pos = 'root'
        orig_head_pos = 'root'

    features.append(head_pos)  # POS of the head of source token
    features.append(L1_upos[L1_token_num])  # POS of the source token
    features.append(head_relation)  # relation between the source token and head
    if not stopWord(orig_head_pos, lemma_head_pos, head_pos, stopwords):
        features.append(lemma_head_pos)
    else:
        features.append('NA')

    if args.use_wsd:
        features.append(wsd_sense)  # WSD of the source token

    token_nums.append(head_source_token - 1)
    token_nums.append(L1_token_num)
    token_nums.append(head_source_token - 1)
    token_nums.append(L1_token_num)
    if args.use_wsd:
        token_nums.append(L1_token_num)

    ##take either 6 dependen
    # Get pos, deprel if the ambig token is head or dep
    num_pos_filled = 6  # Total context features
    dependents = set()
    # consider context +/- 3 tokens surrounding the ambiguous word
    neighbors = set()
    for token_num in L1_indices:
        left_token_index = max(0, token_num - 3)
        right_token_index = min(len(L1_sentence) - 1, token_num + 3)
        for token_index in range(left_token_index, right_token_index):
            if token_index not in L1_indices:
                neighbors.add(token_index)

    for token_index in neighbors:

        if num_pos_filled == 0:
            break
        if token_index in dependent_token_indices:
            dependents.add(token_index)

        if token_index >= len(L1_sentence):
            print(L1_sentence, token_index, len(L1_upos))

        features.append(L1_upos[token_index])  # POS of the depdendent
        token_nums.append(token_index)

        lemma_ = L1_lemma[token_index]
        orig_token_ = L1_sentence[token_index]
        if not stopWord(orig_token_, lemma_, L1_upos[token_index], stopwords):
            features.append(lemma_)
        else:
            features.append('NA')

        token_nums.append(token_index)
        num_pos_filled -= 1

    if num_pos_filled > 0:
        for token_index in dependent_token_indices:
            if token_index not in dependents and num_pos_filled > 0:
                features.append(L1_upos[token_index])
                lemma_ = L1_lemma[token_index]
                orig_token_ = L1_sentence[token_index]
                if not stopWord(orig_token_, lemma_, L1_upos[token_index], stopwords):
                    features.append(lemma_)
                else:
                    features.append('NA')
                num_pos_filled -= 1

                token_nums.append(token_index)
                token_nums.append(token_index)

    while num_pos_filled > 0:
        features.append('NA')
        features.append('NA')

        token_nums.append(-1)
        token_nums.append(-1)
        num_pos_filled -= 1

    # collocation features, bi-gram features of lemma
    stop_words_removed = []
    context_position = []
    original_index = []
    for token_index in range(len(L1_sentence)):
        if token_index == L1_token_num:
            lemma_ = L1_lemma[token_index]
            stop_words_removed.append(lemma_)
            context_position.append(0)
            original_index.append(token_index)
            continue
        # lemma_token = extractLemma(token_index, align_wsd_token[token_index])
        lemma_token = L1_lemma[token_index]
        token_ = L1_sentence[token_index]
        if not stopWord(token_, lemma_token, L1_upos[token_index], stopwords):
            stop_words_removed.append(lemma_token)
            context_position.append(token_index - L1_token_num)
            original_index.append(token_index)

    # consider collocation windoe size of 2, since we have removed stop words, limit = 2*2 = 4
    num_filled = 0
    token_index = context_position.index(0)
    for index in range(token_index - 2, token_index + 3):
        if index == token_index:
            continue
        if index >= 0 and index < len(context_position) and index + 1 < len(context_position):
            set_context = (stop_words_removed[index], stop_words_removed[index + 1])
            features.append(set_context)
            token_nums.append(str(original_index[index]) + "@" + str(original_index[index + 1]))
            num_filled += 1

    while 4 - num_filled > 0:
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
    if len(tgt_lemma) == 0:
        return False, -1
    for alt in label2idindividual.keys():
        if alt == tgt_word:
            return True, label2idindividual[alt]
    for tgt_token in tgt_lemma:
        for alt in label2idindividual.keys():
            if alt == tgt_token:
                return True, label2idindividual[alt]
    return False, -1


def extract_sentences(source_words, source_pos, original_src_sentence, original_tgt_translation, alignments_src,
                      alignments_tgt,
                      src_lemma, source_heads, source_deprels, uposes,
                      tgt_lemma, tgt_heads, tgt_deprels, tgt_uposes,
                      src_wsd_input, sent_num):
    try:

        if args.use_wsd:
            align_wsd_token = alignWSD(original_src_sentence, src_lemma, src_wsd_input, sent_num)

        for token_num, orig_token in enumerate(original_src_sentence):
            pos = uposes[token_num]
            lemma_token = src_lemma[token_num]
            tgt_words, tgt_indices, tgt_indices_string = getTargetAlignedTokens(sent_num, tgt_lemma, token_num,
                                                                                alignments_src)
            if args.reverse:
                L1_words = tgt_words
                L2_words = lemma_token
                L2_Lemmas = src_lemma

            else:
                L1_words = lemma_token
                L2_words = tgt_words
                L2_Lemmas = tgt_lemma

            if isValid(source_words, L1_words) and pos == source_pos:  # (lemma, pos) present in the required words
                isvalid, label = isValidTgt(L2_words, label2idindividual, L2_Lemmas)
                if isvalid:  # Tgt word in the required words
                    label_name = id2label[label]
                    # extract the features for this data sample from the source sentence
                    features, token_nums = extract_features(sent_num, token_num, tgt_indices, align_wsd_token,
                                                            original_src_sentence, uposes, source_heads, source_deprels,
                                                            src_lemma,
                                                            original_tgt_translation, tgt_uposes, tgt_heads,
                                                            tgt_deprels, tgt_lemma)

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
                    df_.to_csv(args.features + ".token.nums", mode='a',
                               header=not os.path.exists(args.features + ".token.nums"))

    except Exception as e:
        # import pdb;pdb.set_trace()
        print(f'Error in {sent_num}', e)


if __name__ == "__main__":
    input_dir = os.path.dirname(args.input)
    input_dir = "./"
    source_pos_target = readInput(args.input_words)

    # create training
    for input_obj in source_pos_target:
        source_words = input_obj.source_words
        source_pos = input_obj.pos
        target_words = input_obj.target_words
        print(f"Processing {target_words[0]} {source_pos} -- {target_words}")
        word_pos = target_words[0] + "_" + source_pos
        args.features = input_dir + "/" + word_pos + "/" + word_pos + ".features"
        print(f"Outputting features in {args.features}")
        os.system(f'rm -rf {input_dir}/{word_pos}/')
        os.system(f'mkdir -p {input_dir}/{word_pos}/')
        sent_num = 0

        filenames = [args.input, args.alignments, args.source_analysis, args.target_analysis, args.wsd, args.orig_input]
        gens = [gen_line(n) for n in filenames]

        # Create the target_classes
        label2id, label2idindividual = {}, {}
        for target_word in target_words:
            target_word = target_word.lstrip().rstrip()
            label2id[target_word] = len(label2id)
            for tgt_word in target_word.split("/"):
                label2idindividual[tgt_word] = label2id[target_word]
        id2label = {v: k for k, v in label2id.items()}
        columns = ['head_pos', 'pos', 'deprel', 'head_lemma']
        if args.use_wsd:
            columns += ['wsd']
        columns += ['dep_pos', 'dep_lemma'] * 6 + \
                   ['lemma-bigram'] * 4 + \
                   ['orig_sentence', 'tgt_translation', 'label', 'source_word', 'tgt_word', 'sent_num']

        total_features = len(columns)
        for input, alignment, source_analysis, target_analysis, wsd_file, original_input in zip(*gens):
            original = input.lower().strip().split(" ||| ")  # Cleaned input
            orig_source, orig_tgt = original[0].split(), original[1].split()

            original_input = original_input.lower().strip().split(" ||| ")  # Original input may have special characters
            original_source, original_target = original_input[0].split(), original_input[1].split()

            alignments_src, alignments_tgt = parseAlignment(alignment.strip().replace("p", "-").split())
            source_lemmas, source_heads, source_deprels, source_feats, source_uposes, source_orig = parseAnalysis(
                source_analysis)
            tgt_lemmas, tgt_heads, tgt_deprels, tgt_feats, tgt_uposes, tgt_orig = parseAnalysis(target_analysis)

            if source_lemmas is None or tgt_lemmas is None:
                print(f"Skipping {sent_num} for lemma and text mismatch")
                continue

            source_lemmas, source_uposes = parseAnalysisSource(source_lemmas, source_orig, orig_source, sent_num,
                                                               source_uposes)
            tgt_lemmas, tgt_uposes = parseAnalysisSource(tgt_lemmas, tgt_orig, orig_tgt, sent_num, tgt_uposes)

            # tgt_lemmas = parseAnalysisTarget(target_analysis, orig_tgt, sent_num)
            source_wsd = wsd_file.strip().split()

            # Clean original source and skip sentences with <=5 words
            clean_tokens = []
            if args.reverse:
                L1_sent = orig_tgt
            else:
                L1_sent = orig_source

            for token in L1_sent:
                if token not in punctuations:
                    clean_tokens.append(token)
            if len(clean_tokens) < 3 and args.prune:
                sent_num += 1
                continue

            try:
                orig_target, tgt_lemmas, tgt_uposes = alignTgtLemma(orig_tgt, tgt_lemmas, original_target,
                                                                    source_upos=tgt_uposes)
                orig_source, source_lemmas, source_uposes = alignTgtLemma(orig_source, source_lemmas, original_source,
                                                                          source_upos=source_uposes)

                if len(orig_source) != len(source_lemmas) or len(orig_target) != len(tgt_lemmas):
                    print("Lemma mismatch", sent_num)
                    sent_num += 1
                    continue


                assert len(orig_target) == len(tgt_lemmas)
                assert len(orig_source) == len(source_lemmas)

                extract_sentences(source_words, source_pos,  orig_source, orig_target,
                                  alignments_src, alignments_tgt,
                                  source_lemmas, source_heads, source_deprels, source_uposes,
                                  tgt_lemmas, tgt_heads, tgt_deprels, tgt_uposes,
                                  source_wsd,
                                  sent_num)
            except Exception as e:
                print(sent_num, e)

            sent_num += 1
            if sent_num % 100000 == 0:
                print(f"Processed {sent_num}")



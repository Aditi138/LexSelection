import codecs
import argparse
import editdistance
import os
from copy import deepcopy
import unidecode
import spacy
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,
                    default="en-es/eng-spa.ambiguous.txt")
parser.add_argument("--output", type=str,
                    default="en-es/source_target_words")
args = parser.parse_args()


def combine(lines):
    nlp = spacy.load('en_core_web_sm')
    lemmatizer = nlp.vocab.morphology.lemmatizer
    lemma_token_map = defaultdict(list)
    original_map = defaultdict(set)
    for line_num, line in enumerate(lines):
        if line_num % 100 == 0:
            print(f"processed {line_num} words")

        info = line.strip().split(" -->\t")
        token_info = info[0].split(",")
        token = token_info[0].lstrip().rstrip().lower()
        pos = token_info[1].lstrip().rstrip()

        if pos == "NUM":  # Avoide numeral
            continue
        lemma = lemmatizer(token, pos)
        for l in lemma:
            lemma_token_map[(l , pos)].append(line_num)
            original_map[(l, pos)].add(token)

    return lemma_token_map, original_map

def getTargetTokens(lines, line_numbers):
    tgt_tokens_info = []
    for line_num in line_numbers:
        line = lines[line_num]
        info = line.strip().split(" -->\t")
        tgt_tokens_info += info[-1].split(", ")

    return tgt_tokens_info

if __name__ == "__main__":
    with codecs.open(args.input, 'r', encoding='utf-8') as fin, codecs.open(args.output, 'w', encoding='utf-8') as fout:
        target_dictionary = {}
        lines = fin.readlines()

        #combine lines having the same lemma
        lemma_token_map, original_map = combine(lines)
        freq_lemma_src_tgt = defaultdict(lambda:0)
        lemma_src_tgt = {}
        for (lemma,pos), line_numbers in lemma_token_map.items():
            lemma=lemma.lower()
            token = lemma
            tgt_tokens_info = getTargetTokens(lines, line_numbers)
            original_tokens = original_map[(lemma, pos)]

            tgt_tokens_set = set()
            normalized_tokens = []
            tgt_tokens = []
            freq_tokens = defaultdict(lambda :0)
            for tgt_token in tgt_tokens_info:
                tgt = tgt_token.split(" ; ")[0].split("=")[0].lstrip().rstrip()
                freq_tokens[tgt] += int(tgt_token.split(" ; ")[-1])
                if "..." in tgt or len(tgt.split()) > 1 or tgt in original_tokens: #Skip for multi-word translations and have same translation as source
                    continue
                tgt_tokens_set.add(tgt)

            for tgt in tgt_tokens_set:
                tgt_tokens.append(tgt)
                normalized_tokens.append(unidecode.unidecode(tgt))

            if len(tgt_tokens) < 2:
                continue
            sorted_normalized_tokens, sorted_tokens = zip(*sorted(zip(normalized_tokens, tgt_tokens)))
            tgt_tokens = sorted_tokens

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
                    if len(prefix) >= t  or un_cur_token in un_prev_token or un_prev_token in un_cur_token: #combine these classes together\
                        same_class = True
                        break

                if same_class:
                    prev_tokens[class_index].append(cur_token)
                else:
                    class_index += 1
                    prev_tokens.append([cur_token])


            if len(prev_tokens) > 1: #After compression there are mutliple tokens:
                output = []
                total_count = 0
                for tgt_tokens in prev_tokens:
                    representation = "/".join(tgt_tokens)
                    count = 0
                    for t in tgt_tokens:
                        count += freq_tokens[t]
                        total_count += freq_tokens[t]
                    if count > 50:  #A target token should have more than 50 sentences
                        output.append(representation + " =  " + str(count))

                if len(output) > 1:
                    #fout.write(token + "," + pos + " --> " + ",".join(output) + "\n")
                    freq_lemma_src_tgt[(token,pos)] = total_count
                    lemma_src_tgt[(token,pos)] = output
        
        #Sort the src-tgt words by freq
        sorted_freq = sorted(freq_lemma_src_tgt.items(), key=lambda kv:kv[1], reverse=True)
        for (key, total) in sorted_freq:
            (token, pos) = key
            output = lemma_src_tgt[key]
            fout.write(token + "," + pos + " --> " + ",".join(output) + "\n")






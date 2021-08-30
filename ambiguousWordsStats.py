import argparse
import spacy
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='eng-spa.ambiguous.wsd.filtered.txt')
parser.add_argument("--output", type=str, default="eng-spa.ambiguous.wsd.filtered.lemma.txt")
args = parser.parse_args()

#filtering exrtacted ambiguous words if lemmatizer had too many issues or class imbalance
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
    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        lines = fin.readlines()
        print(f"Total ambiguous words : {len(lines)}")
        skipped = 0
        for line in lines:
            info = line.strip().split(" --> ")
            lemma, pos = info[0].split(",")[0], info[0].split(",")[1]
            tgt_tokens = info[-1].split(",")
            skip= False
            total = 0
            num_tgt = {}
            for tgt_token in tgt_tokens:
                lemme_values = len(tgt_token.split("/"))
                num_tgt[tgt_token.split(" = ")[0]] = int(tgt_token.split(" = ")[-1])
                total += int(tgt_token.split(" = ")[-1])
                if lemme_values > 4:
                    skip=True

            if skip:
                skipped += 1
                tgt_tokens = ",".join(tgt_tokens)
                print(f"Skipped token: {lemma} {pos}, target tokens: {tgt_tokens} due to lemma")
                continue
            #else:
            #    fout.write(line)

            skip = False
            for tgt_token, value in num_tgt.items():
                p = value * 1.0/total
                if p > 0.80:
                    skip = True
           
            if skip:
                skipped += 1
                tgt_tokens = ",".join(tgt_tokens)
                print(f"Skipped token: {lemma} {pos}, target tokens: {tgt_tokens} due to class imbalance")
                continue
            else:
                fout.write(line)

        print(f"Skipped total: {skipped}")

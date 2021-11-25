import argparse
from collections import defaultdict
import codecs
import math
from utils import *
import stanza


parser = argparse.ArgumentParser()
parser.add_argument("--orig_input", type=str,
                    default="eng-spa")
parser.add_argument("--input", type=str,
                    default="eng-spa")
parser.add_argument("--alignments", type=str,
                    default="eng-spa.pred")
parser.add_argument("--output", type=str,
                    default="output")
parser.add_argument("--source_analysis", type=str,
                    default="eng-spa.analysis")
parser.add_argument("--target_analysis", type=str,
                    default="eng-spa.spa.analysis")
parser.add_argument("--wsd", type=str, default="eng-spa.wsd")
parser.add_argument("--use_wsd", action="store_true", default=False, help="If you have WSD for L1, set it to True")
parser.add_argument("--entropy_threshold", type=float, default=0.60)
parser.add_argument("--freq_threshold", type=int, default=5)
parser.add_argument("--target_freq_threshold", type=int, default=5)
parser.add_argument("--langs", type=str, default='en-es')
parser.add_argument("--reverse", action="store_true", default=False, help="If true then L1 is tgt language and L2 is the src language")
args = parser.parse_args()


def computeEntropy(srctgt_tokens, src_tokens):
    entropy = {}
    for source_token, value in src_tokens.items():
        if source_token not in srctgt_tokens:
            continue
        if value < args.freq_threshold:  # Remove source tokens which occur less than 5 times
            continue
        entropy[source_token] = defaultdict(lambda: 0.0)

        if len(srctgt_tokens[source_token]) == 0: # Remove source tokens which have only target translation
            continue

        for tgt_token, tgt_value in srctgt_tokens[source_token].items():
            if tgt_value < args.target_freq_threshold:  # remove tgt translations which occur less than 5 times (alignment errors)
                continue
            entropy[source_token][tgt_token] = tgt_value * 1.0 / value
    return entropy

def checkMultipleAlignments(tgt_token, target_source_tokens):  # If the target word is multiple source words
    source_words = target_source_tokens[tgt_token]
    alignments = []
    for source_word, val in source_words.items():
        if val < 20: #if the source word is aligned less than 5 times with the target word, this is not included in the multiple alignments
            continue
        alignments.append(source_word)
    if len(alignments) > 3: # if a target word is aligned to multiple source words then we don't want it
        print(tgt_token, alignments)
        return True
    return False


if __name__ == "__main__":
    #Assume the data is pre-cleaned
    filenames = [args.input, args.alignments, args.source_analysis, args.target_analysis, args.orig_input]
    gens = [gen_line(n) for n in filenames] + [gen_line(args.wsd)]


    # Accumulate the counts for source, target
    source_target_tokens = {}
    source_tokens = defaultdict(lambda: 0)
    target_source_tokens = {}
    source_pos_target_tokens = {}
    source_pos_tokens = defaultdict(lambda: 0)
    sent_num=0
    source_pos_target_sent_numbers = {}
    source_pos_wsd_sense = {}

    for input, alignment, source_analysis, target_analysis, original_input, wsd_file in zip(*gens):
        original = input.strip().split(" ||| ") #Cleaned input
        orig_source, orig_target = original[0].lower().split(), original[1].lower().split()

        original_input = original_input.lower().strip().split(" ||| ") #Original input may have special characters
        original_source, original_target = original_input[0].split(), original_input[1].split()

        alignments_src, alignments_tgt = parseAlignment( alignment.strip().replace("p","-").split())

        if args.langs in ['en-es', 'es-en']: #If the syntactic analysis has only 5 cols: lemma, head, deprel, feat, upos
            source_lemmas, source_heads, source_deprels, source_feats, source_uposes = readAnalysis(source_analysis)
            source_orig = orig_source
        else: #If syntactic analysis has either two cols lemma, text or six columns lemma, head, deprel, feat, upos, text
            source_lemmas, source_heads, source_deprels, source_feats, source_uposes, source_orig = parseAnalysis(source_analysis)

        if args.langs in ['en-el', 'el-en', 'en-es', 'es-en']:#if the target analysis contains only lemma, text
            tgt_lemmas = parseAnalysisTarget(target_analysis, orig_target, sent_num)
            tgt_uposes = None

        else: #If the target analyis contains all lemma, head, deprel, feat, upos, text, although we only use the lemma from the target
            tgt_lemmas, _, _, _, _, tgt_orig = parseAnalysis(target_analysis)

        if source_lemmas is None or tgt_lemmas is None:
            print(f"Skipping {sent_num} for lemma and text mismatch")
            continue

        source_lemmas, source_uposes = parseAnalysisSource(source_lemmas, source_orig, orig_source, sent_num, source_uposes)
        if tgt_uposes:
            tgt_lemmas, tgt_uposes = parseAnalysisSource(tgt_lemmas, tgt_orig, orig_target, sent_num, tgt_uposes)

        try:

            orig_target, tgt_lemmas, _ = alignTgtLemma(orig_target, tgt_lemmas, original_target)
            orig_source, source_lemmas, source_uposes = alignTgtLemma(orig_source, source_lemmas, original_source, source_upos=source_uposes)

            if len(orig_source) != len(source_lemmas) or len(orig_target)!=len(tgt_lemmas):
                print("Lemma mismatch", sent_num)
                sent_num += 1
                continue
            if len(source_lemmas) != len(source_uposes):
                print("UPOS mismatch", sent_num)
                sent_num += 1
                continue

            assert len(orig_target) == len(tgt_lemmas)
            assert len(orig_source) == len(source_lemmas)
            if args.use_wsd:
                source_wsd = wsd_file.strip().split()
                align_wsd_token = alignWSD(orig_source, source_lemmas, source_wsd, sent_num)

            for token_num, token in enumerate(source_lemmas):
                tgt_words = []
                tgt_indices = alignments_src[token_num]
                tgt_indices.sort()
                source_pos_word = source_uposes[token_num]
                token = source_lemmas[token_num]
                if args.use_wsd:
                    #gets the wsd from the L1 token,
                    wsd_sense = extractSense(token_num, align_wsd_token[token_num], token)

                if len(tgt_indices) > 1:
                    first_word = tgt_lemmas[0]
                    last_word = tgt_lemmas[-1]
                    if isContiguous(tgt_indices):
                        for index in tgt_indices:
                            tgt_token = tgt_lemmas[index]
                            tgt_words.append(tgt_token)
                        tgt_words = " ".join(tgt_words)
                    else:
                        tgt_words = first_word + " ... " + last_word  # non-contiguous tokens

                elif len(tgt_indices) == 1:
                    tgt_words = tgt_lemmas[tgt_indices[0]]

                if len(tgt_words) > 0:
                    if args.reverse:
                        L1_words = tgt_words
                        L2_words = token
                    else:
                        L1_words = token
                        L2_words = tgt_words

                    if (L1_words, source_pos_word) not in source_pos_target_tokens:
                        source_pos_target_tokens[(L1_words, source_pos_word)] = defaultdict(lambda: 0)
                        source_pos_target_sent_numbers[(L1_words, source_pos_word)] = defaultdict(list)
                        source_pos_wsd_sense[(L1_words, source_pos_word)] = {}
                    if L1_words not in source_target_tokens:
                        source_target_tokens[L1_words] = defaultdict(lambda: 0)

                    source_target_tokens[L1_words][L2_words] += 1
                    source_tokens[L1_words] += 1
                    source_pos_target_tokens[(L1_words, source_pos_word)][L2_words] += 1
                    source_pos_tokens[(L1_words, source_pos_word)] += 1
                    source_pos_target_sent_numbers[(L1_words, source_pos_word)][L2_words].append(str(sent_num))


                    if L2_words not in target_source_tokens:
                        target_source_tokens[L2_words] = defaultdict(lambda: 0)

                    target_source_tokens[L2_words][L1_words] += 1

                    if args.use_wsd and wsd_sense != 'NA':
                        #print(token, source_pos_wsd_sense, tgt_words, wsd_sense)
                        source_pos_wsd_sense[(L1_words, source_pos_word)][L2_words][wsd_sense] += 1

            sent_num += 1
            if sent_num % 1e+6 == 0:
                print(f"Processed {sent_num}")

        except Exception as e:
            print(sent_num, e)
            sent_num += 1


    print("Combine lemmas using edit distance")
    source_pos_target_tokens, _ = filter_words(source_pos_target_tokens, target_source_tokens, args.target_freq_threshold, fileprefix=args.langs, combine_lemmas=False)

    print("Computing entropy ....")
    entropy = computeEntropy(source_pos_target_tokens, source_pos_tokens)
    source_tgt_words_with_no_wsd = 0

    # Output ambiguous words having freq of source words > args.freq_threhold & entropy (H(w_x) > args.entropy_threshold (log_e)
    with codecs.open(args.output, 'w') as fout:
        ambiguous_words = {}
        for token, tgt_tokens in entropy.items(): #token == (lemma, pos)
            H = 0.0
            translations = {}
            wsd_sense_tgt_tokens = defaultdict(set)
            for tgt_token, prob in tgt_tokens.items():

                H += -prob * math.log(prob)
                translations[tgt_token] = -prob * math.log(prob)

                if args.use_wsd:
                    if tgt_token in source_pos_wsd_sense[token]:

                        wsd_senses = source_pos_wsd_sense[token][tgt_token]
                        if len(wsd_senses) > 0:  # non-NA wsd senses, sort the sense
                            sorted_wsd_sense = sorted(wsd_senses.items(), key=lambda kv: kv[1], reverse=True)
                            for (wsd_sense, _) in sorted_wsd_sense:
                                wsd_sense_tgt_tokens[tgt_token].add(wsd_sense)
                        else:
                            source_tgt_words_with_no_wsd += 1

            if H < args.entropy_threshold:
                continue

            if args.use_wsd: #Remove L1 words which have different word sense for the different L2 translation
                freq = defaultdict(lambda: 0)
                if len(wsd_sense_tgt_tokens) == len( tgt_tokens):  # All the target tokens have wsd sense identified only then we should perform this filtration
                    first = True
                    inter = set()
                    for tgt_token, wsd_value_set in wsd_sense_tgt_tokens.items():
                        if first:
                            inter = wsd_value_set
                            first = False
                        else:
                            inter = inter & wsd_value_set

                    if len(inter) == 0:  # All target tokens have the distinct wsd_value_set
                        print(f"Skipped token:{token[0]} due to wsd filtering")
                        continue

            sorted_translations = sorted(translations.items(), key=lambda kv: kv[1], reverse=False)
            tgt_tokens = []
            tgt_words = []
            for (tgt_token, H) in sorted_translations:
                tgt_tokens.append(tgt_token + "= " + str(H) + " ; " + str(source_pos_target_tokens[token][tgt_token]))
            if len(tgt_tokens) > 0:
                (token, pos) = token
                if pos not in ['NUM', 'PROPN', 'PUNCT', 'SYM', 'X']:
                    fout.write(token + ", " + pos + " -->\t" + ", ".join(tgt_tokens) + "\n")

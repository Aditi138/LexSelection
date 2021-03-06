First, we describe the method for identifying semantic subdivisions followed by the methods to extract human-understandable descriptions to help understand  when one lexical distinction to be used over the other.

# Identifying Semantic Subdivisions
*** Update ***
We have added a more generic function `extractAmbiguousWords_allow_mwt` which allows you to extract semantic subdivisions in either direction i.e. learn Spanish
from English or learn English from Spanish.
Further, we make the reliance of WSD features optional as for languages other than English they might be harder to obtain.

Assuming the parallel data from English to target language has been lemmatized, POS tagged and parsed, run the following to extract English focus words:
```     DIR=/data/en-es/
        python extractAmbiguousWords.py \
        --orig_input $DIR/eng-spa.clean \
        --input $DIR/eng-spa.clean \
        --alignments $DIR/eng-spa.pred \
        --source_analysis $DIR/eng-spa.analysis.clean \
        --target_analysis $DIR/eng-spa.spa.lemma \
        --wsd $DIR/eng-spa.wsd \
        --output $DIR/eng-spa.ambiguous.txt \
        --freq_threshold 10 \
        --target_freq_threshold 20 \
        --sent_numbers $DIR/sent_numbers.txt

        python filterWords.py \
           --input $DIR/eng-spa.ambiguous.txt \
           --output $DIR/eng-spa.edit.txt

        python ambiguousWordsStats.py \
            --input $DIR/eng-spa.edit.txt \
            --output $DIR/$DIR/eng-spa.filtered.txt
 ```
 where ```--orig_input``` and ```--input``` refers to the cleaned parallel data, ```--alignments``` refer to the word alignments, ```--source_analysis``` refers to the lemmatized, POS tagged and parsed English data,
 ```--target_analysis``` refers to the lemmatized Spanish data, ```--wsd``` refers to the word sense disambiguation model output for English portion of the data and ```--output``` is the
list of focus words returned after running the model. 
The script ```filterWords.py``` runs edit-distance based post-processing. This combines lexical choices which have possibly  same lemma, however this sometimes leads to more than 5 words being conflated in the single lexical choice, which possibly hints at 
asignificant amount of lemmatization issues. 
Therefore, we run a cleaning script ```ambiguousWordsStats.py``` which further filters focus words which have >5 lemmas being conflated together and an addional frequency filter to avoid words that have class imbalance.

# Data
We have provided the semantic subdivisions for English-Spanish (```data/es.words```) and  English-Greek (```data/el.words```). The format of the data is 
``` en-word,POS,tgt-word1;tgt-word2;... ``` where the en-word and tgt-word* are lemmatized forms. In some cases multiple lemmas are conflated together  to fix lemmatizatomn issues. An example:
``` wall,NOUN,muralla/muro/muros;pared/pared??n ``` where ``` muro ``` and ``` pared ``` are the two lexical choices for ``` wall ```. 

# Training a Lexical Selection Model
1. Feature extraction: extract features and train/test split for each focus word.
```     DIR=/data/en-es/


        python extract_train_dev_test_index.py \
        --orig_input $DIR/eng-spa.clean \
        --input $DIR/eng-spa.clean \
        --alignments $DIR/eng-spa.pred \
        --source_analysis $DIR/eng-spa.analysis.clean \
        --target_analysis $DIR/eng-spa.spa.lemma \
        --wsd $DIR/eng-spa.wsd \
        --input_words $DATA/source_target_words \
        --prune

    python extract_features.py \
        --input /data/en-es/ \
        --word language_NOUN
```
where ```--input_words``` specifies the focus word and its lexical choices for which we want to extract the data. This outputs a folder per each focus word in the $DIR
Sample data format is ```data/```


2. Train a lexical model for each focus word
```
WORDS="language_NOUN ticket_NOUN wall_NOUN vote_NOUN oil_NOUN driver_NOUN farmer_NOUN computer_NOUN servant_NOUN figure_NOUN pill_NOUN wave_NOUN"
for w in $WORDS
do
	python -u train.py \
		--input ~/data/en-es \
		--word $w  2>&1 | tee $w.svm.log

done
```
This returns the model accuracy and also prints the top-20 features for each lexical choice.

3. After training model extract human readable rules from the trained model
```
python -u humanReadableRules.py \
		--input ~/data/en-es \
		--word $w \
		--feature_map ./feature_map
```
This outputs a .json file preparing the data for human annotation with human-readable rules.

# Baseline Models
1. For training BERT models run 
``` 
python computeBertBaselineAccuracy.py \
          --input ~/data/en-es  \
	  --word wall_NOUN 
```
  This will print the test accuracy for BER-baseline.
  
2. For training decision tree run
```
 python -u train.py \
	--input ~/data/en-es \
	--word $w  --use_dtree 2>&1 | tee $w.dtree.log
```

### References
If you make use of this software for research purposes, we will appreciate citing the following:
```
@inproceedings{chaudhary21emnlp,
    title = {When is Wall a Pared and when a Muro?:Extracting Rules Governing Lexical Selections},
    author = {Aditi Chaudhary and Kayo Yin and Antonios Anastasopoulos  and Graham Neubig},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    address = {Online},
    month = {November},
    year = {2021},
    url = {https://arxiv.org/abs/2109.06014}
}
```

### Contact
For any issues, please feel free to reach out to `aschaudh@andrew.cmu.edu`.

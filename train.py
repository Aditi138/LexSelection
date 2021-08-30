import pandas as pd
import numpy as np
np.random.seed(1)
import argparse
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import  GridSearchCV
import os
import sklearn
from copy import deepcopy
from utils import removeStopWordsPunctuations, calculate_top_contributors, plot_coefficients, f_importances, plot_coefficients_label
import xgboost as xgb
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="folder")
parser.add_argument("--word", type=str, default='worker_NOUN')
parser.add_argument("--use_xgboost", action="store_true", default=False)
parser.add_argument("--use_dtree", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=1001)
parser.add_argument("--load_params", action="store_true", default=False)
parser.add_argument("--best_params", type=str, default="2,2,0.001,257") #3,1,1,357
parser.add_argument("--prune", action="store_true", default=True)
parser.add_argument("--print", action="store_true", default=False)
parser.add_argument("--split", action="store_true", default=False)
parser.add_argument("--topk", type=int, default=8)
parser.add_argument("--best_params_list", type=str)
parser.add_argument("--test_file", type=str, default=None)
parser.add_argument("--num", type=int, default=-1)
args = parser.parse_args()

def split_train_test(input_file):
    train_input = input_dir + f'{args.word}.new.train.features'
    test_input = input_dir + f'{args.word}.new.test.features'
    required_columns_file = input_dir + f'{args.word}.col.names'

    if os.path.exists(train_input):
        print(f"Reading existing train/test from {train_input}")
        all_train, all_test = pd.read_csv(train_input, sep=","), pd.read_csv(test_input, sep=",")
        new_columns = all_train.columns.tolist()

        #Remove duplicates from the all_train
        train_sents = all_train['orig_sentence']
        test_sents = all_test['orig_sentence']
        intersection = set(train_sents) & set(test_sents)
        for sent in intersection:
            indexNames = all_train[all_train['orig_sentence'] == sent].index
            all_train.drop(indexNames, inplace=True)
        print("Removed", len(intersection))

    else:
        print(f'Creating train/test split: {input_file}')
        data = pd.read_csv(input_file, sep=',')
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

        with open(required_columns_file, 'w') as fout:
            fout.write("\t".join(new_columns) + "\n")
            fout.write("\t".join(label_columns) + "\n")
    print(f"Train {all_train.shape}, Test {all_test.shape}.")
    return all_train, all_test, new_columns

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

def printTreeWithExamplesPDF(tree_rules, feature_names, class_names ):
    lines = tree_rules.split("\n")
    rules = []
    leaves = []
    for line in lines:
        if "<=" in line:
            info = line.split("<=")
            depth = info[0].count("|")
            feature_info = info[0].split('--- ')[-1].lstrip().rstrip()
            if len(rules) == 0:
                rules.append([(depth,feature_info, 0)])
            else:
                current_rules = rules[-1]
                prev_depth, prev_feature, prev_value = current_rules[-1]
                if depth > prev_depth:
                    rules[-1].append((depth, feature_info, 0))
                else:# Keep popping till you are at the same depth Starting a new branch
                    new_rules = deepcopy(rules[-1])
                    while prev_depth != depth:
                        new_rules.pop()
                        prev_depth, prev_feature, prev_value = new_rules[-1]
                    new_rules.pop() # removing the last branch
                    new_rules.append((depth, feature_info, 0))
                    rules.append(new_rules)

        elif 'class' in line:
            class_ = line.strip().split("---")[-1].lstrip().rstrip()
            leaves.append(class_)

        elif ">" in line:
            info = line.split(">")
            depth = info[0].count("|")
            feature_info = info[0].split('--- ')[-1].lstrip().rstrip()
            if len(rules) == 0:
                rules.append([(depth,feature_info, 1)])
            else:
                current_rules = rules[-1]
                prev_depth, prev_feature, prev_value = current_rules[-1]
                if depth > prev_depth:
                    rules[-1].append((depth, feature_info, 1))
                else:# Starting a new branch
                    new_rules = deepcopy(rules[-1])
                    while prev_depth != depth:
                        new_rules.pop()
                        prev_depth, prev_feature, prev_value = new_rules[-1]
                    new_rules.pop()  # removing the last branch
                    new_rules.append((depth, feature_info, 1))
                    rules.append(new_rules)


        #print(len(rules))

def train_xgboost(X, y, feature_names, num_classes, test, y_test):
    model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5, gamma=0, subsample=0.8,
                          colsample_bytree=0.8, objective='multi:softprob', num_class= num_classes, silent=True, nthread=1,
                          scale_pos_weight=majority/minority, seed=1001)

    cvFolds ,early_stopping_rounds = 5,50
    xgbParams = model.get_xgb_params()
    xgTrain = xgb.DMatrix(X, label=y)

    if args.load_params:
        print("Loading best params")
        best_params= args.best_params.split(",")
        print("Best params: {}, {}, {}, {}".format(best_params[0], best_params[1], best_params[2],
                                                             best_params[3]))

        xgbParams['max_depth'] = int(best_params[0])
        xgbParams['min_child_weight'] = int(best_params[1])
        xgbParams['gamma'] = float(best_params[2])
        num_boost_round = int(best_params[3])

    else:
        print("Hyperparameter tuning")

        gridsearch_params = [
            (max_depth, min_child_weight, gamma)
            for max_depth in range(2, 4)
            for min_child_weight in range(1, 4)
            for gamma in [0.001, 0.5, 1]
        ]
        min_mlogloss = float("Inf")
        best_params = None

        for max_depth, min_child_weight, gamma in gridsearch_params:
            print("CV with max_depth={}, min_child_weight={}, gamma={}".format(
                max_depth,
                min_child_weight,
                gamma))
            xgbParams['max_depth'] = max_depth
            xgbParams['min_child_weight'] = min_child_weight
            xgbParams['gamma'] = gamma

            cv_results = xgb.cv(xgbParams,
                          xgTrain,
                          num_boost_round= model.get_params()['n_estimators'],
                          nfold=cvFolds,
                          stratified=True,
                          metrics='mlogloss',
                          early_stopping_rounds=early_stopping_rounds,
                          seed=0)

            mean_mlogloss = cv_results['test-mlogloss-mean'].min()
            boost_rounds = cv_results['test-mlogloss-mean'].argmin()
            print("\tmlogloss {} for {} rounds".format(mean_mlogloss, boost_rounds))
            if mean_mlogloss < min_mlogloss:
                min_mlogloss = mean_mlogloss
                best_params = (max_depth, min_child_weight, gamma, len(cv_results))
        print("Best params: {}, {}, {}, mlogloss: {}".format(best_params[0], best_params[1], best_params[2],  min_mlogloss))

        #update best params
        print(best_params)

        xgbParams['max_depth'] = best_params[0]
        xgbParams['min_child_weight'] =best_params[1]
        xgbParams['gamma'] = best_params[2]
        num_boost_round = best_params[3]

    # Fit the algorithm
    best_model = xgb.train(
        xgbParams,
        xgTrain,
        evals=[(test, 'test')],
        num_boost_round= num_boost_round
    )
    # Predict
    dtrainPredictions_prob = best_model.predict(xgTrain)
    dtestPredictions_prob = best_model.predict(test)

    dtrainPredictions = np.argmax(dtrainPredictions_prob, axis=1)
    dtestPredictions = np.argmax(dtestPredictions_prob, axis=1)


    # Print model report:
    print("Train Accuracy : %.4g" % accuracy_score(y, dtrainPredictions))
    print("Test Accuracy : %.4g" % accuracy_score(y_test, dtestPredictions))

    model_file = input_dir  + 'model.raw.txt'
    best_model.dump_model(model_file)

    shap_values = best_model.predict(xgTrain, pred_contribs=True)
    output_file = input_dir + f"/important.datapoints.test.txt"

    incorrect_preds_file = input_dir + f"/incorrect_preds.test"
    print()
    lengths = defaultdict(list)
    print(confusion_matrix(y, dtrainPredictions))
    max_lenght = 0
    for index in range(len(all_test)):
        original_sentence= all_test.iloc[index]["orig_sentence"].split()
        lengths[len(original_sentence)].append(index)
        if len(original_sentence) > max_lenght:
            max_lenght = len(original_sentence)

    required_indices = []
    for length in range(5, max_lenght+1): #lenght >=5
        #print(length, len(lengths[length]))
        required_indices += lengths[length]

    #get imp
    imp_features = best_model.get_score(importance_type='weight')
    imp_features_index = []
    important_feature_names = []
    imp_features_lambda = []
    for f in imp_features.keys():
        imp_features_index.append(feature_names.index(f))
        imp_features_lambda.append(imp_features[f])
        important_feature_names.append(f)

    imp_features_lambda = np.array(imp_features_lambda)
    #imp_features_lambda = imp_features_lambda / sum(imp_features_lambda)  # Normalizing the important features
    important_feature_names = np.array(important_feature_names)
    important_feature_values_, important_feature_names_ = zip(
        *sorted(zip(imp_features_lambda, important_feature_names), reverse=True))
    important_features_output_file = input_dir + f'/important.features'



    preds = dtrainPredictions
    preds_prob = dtrainPredictions_prob

    output_dict =  classification_report(y, dtrainPredictions, target_names=label_list,output_dict=True)
    for key, value in output_dict.items():
        if key in label_list:
            print(key,value['f1-score'])

    if args.print:
        exit(-1)

    with open(output_file, 'w') as fout, open(incorrect_preds_file, 'w') as fin, open(important_features_output_file, 'w') as ffeat:
        selected_true_probabilities, selected_indices = [], []
        print(f"Top features outputted to {output_file} and {important_features_output_file}")
        features_label_id_in_order = defaultdict(list)
        common_features = defaultdict(set)
        for label, id in label2id.items():
            true = 0
            imp_feature_values = [0] * len(imp_features_index)
            required_indices = np.where(y == id)[0]  # Examples where true_label == id
            data_points_per_index = defaultdict(set)
            data_points_per_feature = defaultdict(list)
            for i in required_indices:
                pred = preds[i]
                if pred == y[i]: #IF model pred is true
                    datavalues = []
                    for f in important_feature_names:
                        datavalues.append(X.iloc[i][f])
                    datavalues = np.array(datavalues)
                    #datavalues = X.iloc[i, imp_features_index].values
                    true_prob = preds_prob[i][id]
                    non_zero_index = datavalues > 0
                    non_zero_names = important_feature_names[non_zero_index]

                    datavalues = np.array([1 if i == 1 else -1.0 for i in datavalues])
                    # total_sum = np.sum(datavalues)
                    # if total_sum > 0:
                    #     datavalues = datavalues / total_sum
                    imp_feature_values += np.multiply(imp_features_lambda, datavalues) #Checking the number of examples having active features for each label from the imp features

                    data_points_per_index[i] = non_zero_names
                    for f in non_zero_names:
                        data_points_per_feature[f].append(i)
                    true += 1
                    selected_true_probabilities.append(true_prob)
                    selected_indices.append(i)


            #Sort the imp_feature_values
            important_feature_values_, important_feature_names_  = zip(*sorted(zip(imp_feature_values, important_feature_names), reverse=True))
            important_feature_per_label, important_feature_per_label_value = [], []
            total_sum = np.sum(important_feature_values_)
            t = 0.8 * total_sum #Taking top features which account of 80% of all feature importance
            sum_ = 0
            print(label)

            for feat_name, feat_value in zip(important_feature_names_, important_feature_values_):
                important_feature_per_label.append(feat_name)
                important_feature_per_label_value.append(feat_value)
                sum_ += feat_value
                features_label_id_in_order[id].append((feat_name, feat_value))
                common_features[feat_name].add(id)
                print(feat_name, feat_value)

                if  len(features_label_id_in_order[id]) > 10:
                    break


            print(
                f"Extracted feature importance for label: {label}, correct: {true} of datapoints: {len(required_indices)}, train  acc: {true / len(required_indices)}")
            print()



        #Write feature names, currently not removing common features.
        to_remove_features = set()
        for common_feature, ids in common_features.items():
            if len(ids) == len(label2id): #These features are present for all classes, we should remove them
                to_remove_features.add(common_feature)
        to_remove_features = set()

        for label, id in label2id.items():
            print(label)
            ffeat.write(str(id) + "," + label + ";")
            all_features = features_label_id_in_order[id]
            to_retain_features = []
            for (feat_name, feat_value) in all_features:
                if feat_name in to_remove_features:
                    continue
                else:
                    to_retain_features.append(feat_name)

            if len(to_retain_features) == 0: #The label had features common to all features then add those back
                for (feat_name, feat_value) in all_features:
                    to_retain_features.append(feat_name)

            to_retain_features = np.array(to_retain_features[:args.topk])
            for feat_name in to_retain_features:
                print(feat_name)
                ffeat.write(feat_name + "~~~")
            ffeat.write("\n")
            print()

            # Randomly select upto 20 examples for each label
            required_indices = np.where(y_test == id)[0]  # Examples with true label == id
            # intersection = set(required_indices) & set(y_filtered_indices) #Intersection of ids with lenght >=5

            for i in required_indices:
                datavalues = []
                for f in to_retain_features:
                    datavalues.append(all_test_features.iloc[i][f])
                datavalues = np.array(datavalues)

                feat_names = to_retain_features[datavalues > 0]
                feature_names_intersection, feature_index_intersection = getFeatureIndex(all_test.iloc[i], feat_names,
                                                                                         to_retain_features,
                                                                                         feature_index_data)
                input_sents = all_test.iloc[i]["orig_sentence"] + "\t" + all_test.iloc[i][
                    "tgt_translation"] + "\t" + str(all_test.iloc[i]["source_word"]) + "\t" + str(
                    all_test.iloc[i]["tgt_word"])
                fout.write(
                    str(id) + "\t" + label + "\t" + "~~~".join(feature_names_intersection) + "\t" + "~~~".join(
                        feature_index_intersection) + "\t" + input_sents + "\n")

def train_dtree(train_features, train_output_labels, feature_names, test_features, test_output_labels):

    x_train, y_train, x_test, y_test = train_features , train_output_labels, test_features, test_output_labels
    x,y = x_train, y_train
    cv = 5
    # Create lists of parameter for Decision Tree Classifier
    criterion = ['gini', 'entropy']
    parameters = {'criterion':criterion, 'max_depth':np.arange(3, 7), 'min_impurity_decrease':[1e-3,1e-2, 1e-1],
                  'random_state':[3], 'class_weight':['balanced', None], 'min_samples_leaf':[5]}
    decision_tree = DecisionTreeClassifier()
    # parameters = { 'C': [0.001, 0.01], 'class_weight':['balanced', None], 'random_state':[3]}
    linearsvm  = LinearSVC()
    print("Hyperparameter tuning")
    model = GridSearchCV( decision_tree , parameters, cv=cv)
    model.fit(x, y)

    best_model = model.best_estimator_
    print('CV best accuracy', model.best_score_, model.best_params_)
    dtrainPredictions = best_model.predict(x)
    dtestPredictions = best_model.predict(x_test)

    # Print model report:
    print("Train Accuracy : %.4g" % accuracy_score(y, dtrainPredictions))
    print("Test Accuracy : %.4g" % accuracy_score(y_test, dtestPredictions))




def train(train_features, train_output_labels, feature_names, test_features, test_output_labels):

    x_train, y_train, x_test, y_test = train_features , train_output_labels, test_features, test_output_labels
    x,y = x_train, y_train
    cv = 5
    # Create lists of parameter for Decision Tree Classifier
    # criterion = ['gini', 'entropy']
    # parameters = {'criterion':criterion, 'max_depth':np.arange(3, 7), 'min_impurity_decrease':[1e-3,1e-2, 1e-1], 'random_state':[3], 'class_weight':['balanced', None],
    #               'min_samples_leaf':[5]}
    # decision_tree = sklearn.tree.DecisionTreeClassifier()
    parameters = { 'C': [0.001, 0.01], 'class_weight':['balanced', None], 'random_state':[3]}
    linearsvm  = LinearSVC()
    print("Hyperparameter tuning")
    model = GridSearchCV( linearsvm , parameters, cv=cv)
    model.fit(x, y)

    best_model = model.best_estimator_
    print('CV best accuracy', model.best_score_, model.best_params_)
    dtrainPredictions = best_model.predict(x)
    dtestPredictions = best_model.predict(x_test)

    # Print model report:
    print("Train Accuracy : %.4g" % accuracy_score(y, dtrainPredictions))
    print("Test Accuracy : %.4g" % accuracy_score(y_test, dtestPredictions))

    if args.test_file: #only produce
        print(args.test_file)
        test_data =  pd.read_csv(args.test_file, sep=',')
        all_test_features, all_test_label = test_data[required_columns], label_encoder.transform(test_data[["label"]])

        if args.num > -1:
            all_test_features = all_test_features.head(args.num)
            all_test_label = all_test_label[:args.num]
        dtestPredictions = best_model.predict(all_test_features)
        print("Test Accuracy : %.4g" % accuracy_score(all_test_label, dtestPredictions))
        exit(-1)

    print(confusion_matrix(y, dtrainPredictions))

    output_dict = classification_report(y, dtrainPredictions, target_names=label_list, output_dict=True)
    for key, value in output_dict.items():
        if key in label_list:
            print(key, value['f1-score'])

    #Coefficients per each class
    importance_features = plot_coefficients_label(best_model, feature_names ,label_list, args.word)

    important_features_output_file = input_dir + f'/important.features'

    output_file = input_dir + f"/important.datapoints.test.txt"

    with open(output_file, 'w') as fout, open(important_features_output_file, 'w') as ffeat:
        for label_id, features in importance_features.items():
            label = id2label[label_id]
            ffeat.write(str(label_id) + "," + label + ";")
            for feat_name in features:
                ffeat.write(str(feat_name) + "~~~")
            ffeat.write("\n")

            # Randomly select upto 20 examples for each label
            required_indices = np.where(y_test == label_id)[0]  # Examples with true label == id
            # intersection = set(required_indices) & set(y_filtered_indices) #Intersection of ids with lenght >=5

            for i in required_indices:
                datavalues = []
                features = np.array(features)
                for f in features:
                    datavalues.append(all_test.iloc[i][f])
                datavalues = np.array(datavalues)

                feat_names = features[datavalues > 0]
                feature_names_intersection, feature_index_intersection = getFeatureIndex(all_test.iloc[i], feat_names,
                                                                                         features,
                                                                                         feature_index_data)
                input_sents = all_test.iloc[i]["orig_sentence"] + "\t" + all_test.iloc[i][
                    "tgt_translation"] + "\t" + str(all_test.iloc[i]["source_word"]) + "\t" + str(
                    all_test.iloc[i]["tgt_word"])
                fout.write(
                    str(label_id) + "\t" + label + "\t" + "~~~".join(feature_names_intersection) + "\t" + "~~~".join(
                        feature_index_intersection) + "\t" + input_sents + "\n")

    # tree_rules = export_text(model.best_estimator_, feature_names= feature_names, max_depth=model.best_params_["max_depth"])
    # printTreeWithExamplesPDF(tree_rules, feature_names, label_columns)
    # print(tree_rules)
    print(label2id)

def filteredFeatures(all_features):
    required_features = []
    print("Removing syntactic features ")
    for feat in all_features:
        feat_info = feat.split('__')[0]
        word = args.word.split("_")[0]
        if feat in ['orig_sentence', 'tgt_translation', 'label', 'sent_num', 'source_word', 'tgt_word']:
            continue
        if 'Unnamed' in feat or "UNK" in feat:
            continue
        if feat_info == 'pos' or feat_info == "head_pos" or feat_info == "deprel":
            continue
        #Remove other dependent lemmas (except for head_lemx`ma) which are present in the stopword list
        if "[" in feat or "]" in feat or "<" in feat:
            continue
        if 'lemma' in feat:
            lemma = feat.split('__')[-1]
            if removeStopWordsPunctuations(lemma):
                continue
        if "is_dep" in feat or 'dep_pos' in feat or 'del_rel' in feat:
            continue
        if feat == "lemma__root" or "__root" in feat:
            continue

        required_features.append(feat)
    print(f'Filtering Features, before: {len(all_features)}, after: {len(required_features)}')
    #print(f'new columns: {required_features}')
    return required_features

def getFeatureIndex(test_example, feat_names, important_feature_per_label, df):
    orig_sent = test_example['orig_sentence']
    tgt_sent = test_example['tgt_translation']
    sent_num = test_example['sent_num']

    required_index = df.loc[df['sent_num'] == sent_num].index.tolist()[0]
    required_data = df.iloc[required_index]
    assert required_data['orig_sentence'] == orig_sent

    intersection = set(feat_names) & set(important_feature_per_label)

    feature_names, feature_index = [],[]
    for feature in intersection:

        token_num = required_data[feature]
        feature_names.append(feature)
        feature_index.append(str(token_num))
    return feature_names, feature_index



if __name__ == "__main__":
    input_dir = args.input + f'/{args.word}/'
    input_file = input_dir + f'{args.word}.features'
    token_nums_file = input_dir + f'{args.word}.new.test.features.token.nums'

    all_train, all_test, required_columns = split_train_test(input_file)
    feature_index_data = pd.read_csv(token_nums_file, sep=',')

    if args.split:#Splitting data into train/test
        exit(-1)


    required_columns = filteredFeatures(required_columns)

    #combine all data and report cross-validation scores
    all_data = all_train
    #shuffle training data
    all_data = all_data.sample(frac=1)
    #Get labels
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(all_train[["label"]])
    label2id = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    id2label = {v:k for k,v in label2id.items()}
    print(label2id)


    all_train_features, all_train_label = all_data[required_columns], label_encoder.transform(all_data[["label"]])
    all_test_features, all_test_label = all_test[required_columns], label_encoder.transform(all_test[["label"]])



    counter = Counter(all_train_label)
    total_samples = len(all_data)
    baseline_accuracy_per_class = {}
    minority = 10000000
    majority = -1
    label_list = []
    for class_ in range(len(counter)):
        label_list.append(id2label[class_])
        value = counter[class_]
        if value < minority:
            minority = value
        if value > majority:
            majority = value
        baseline_accuracy_per_class[id2label[class_]] = value * 1.0 / total_samples
        print(f"Baseline score for class {class_}: {id2label[class_]} = {value * 1.0 / total_samples}")


    if args.use_xgboost: #UseXGBoost:
        print("Using XgBoost")
        #Loading existing best params from a file if it exists
        if args.best_params_list:
            with open(args.best_params_list, 'r') as fin:
                for line in fin.readlines():
                    if args.word in line:
                        info = line.strip().split(",")
                        info = [i.lstrip().rstrip() for i in info]
                        args.best_params = ",".join(info[1:])
                        args.load_params = True
                        print('Loading best params', args.best_params)
                        break

        xgtest = xgb.DMatrix(all_test_features, label=all_test_label)
        train_xgboost(all_train_features, all_train_label, required_columns, num_classes=len(counter), test=xgtest, y_test=all_test_label)
    elif args.use_dtree:
        print('Using dTree')
        train_dtree(all_train_features, all_train_label, required_columns, all_test_features, all_test_label)
    else:
        print("Using Sklearn")
        train(all_train_features, all_train_label, required_columns, all_test_features, all_test_label)

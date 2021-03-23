# Modular Programming and scikit-learn
This package contains a POS-tagger that can be trained by the user and used to tag sentences/text files.

## Design
This package contains three python files: `pre_process.py` and `model.py` in the `tagger` folder, and `run.py` in the main folder that is called when running the programe.

### `pre_process.py`
`pre_process.py` defines how dataset is loaded and how features are extratced. In particular, the loaded file can be in either `.txt` format or `.conllu` format. After loading in the data, features are extracted as dictionaries for each token, and `DictVectorizer` is used to transform list of dictionaries into a feature matrix.

**Feature engineering**: compared to the original version illustrated in class, in this assignment more features are added. In particular, informations about the second words both to the left and to the right of the token is extracted and stored in the token's feature dictionary; what's more, the token's lemma is also stored in its feature dictionary.

### `model.py`
`model.py` defines how training data is trained, how the trained the model is saved/loaded, and how validation/test data can be tagged using the trained model.

**Feature engineering**: compared to the original version illustrated in class, in this assignment more training models are added. In particular, a stochastic gradient descent model and a decision tree model have been added.

### `run.py`
`run.py` can be called in command line to (1) train a model, (2) tag a file/sentence using the trained model, and (3) evaluate the accuracy of the trained model. In the following <em>Usage</em> part, more information can be found in terms of what arguments can be passed to `run.py`.



## Usage

### Train a new model
- Pass `train` to the argument `--mode`
- Pass the selected training model (`svm`, `sgd`, or `dt`) to the argument `--model`
- Pass the config file to the argument `--config`


An example:

    % python3 run.py --mode train 
                     --model svm 
                     --config config.yaml
, and the output is:

    Training a model on ./en_ewt/en_ewt-ud-train.conllu.
    Doing 5-fold cross-validation.
    5-fold cross-validation results over training set:

    Fold           Score
    1              0.940
    2              0.935
    3              0.936
    4              0.935
    5              0.937
    Average        0.937
    Fitting model.
    Took 16 seconds.
    Saving model to ./trained_tagger.pickle.

### Tag a file/sentence
- Pass `tag` to the argument `--mode`
- Pass a single sentence or a text file to the argument `--text`
- Pass the config file to the argument `--config`

An example:

    % python3 run.py --mode tag 
                     --text texts/speech_000.txt 
                     --config config.yaml
, and the output is:

    Tagging text using pretrained model: ./en_ewt/en_ewt-ud-train.conllu.
    Loading model from ./trained_tagger.pickle.
    File saved to the same directory as the input file.
where the predicated tags are  saved in a separate text file in the *same directory* as the input file, with the suffix `.tag` (i.e. `my_file.txt` → `my_file.txt.tag`).

### Evaluate the trained model
- Pass `eval` to the argument `--mode`
- Pass a gold file to the argument `--gold`
- Pass the config file to the argument `--config`

An example:

    % python3 run.py --mode eval 
                     --gold ./en_ewt/en_ewt-ud-test.conllu 
                     --config config.yaml
, and the output is:

    Evaluating model on ./en_ewt/en_ewt-ud-test.conllu.
    Loading model from ./trained_tagger.pickle.
                  precision    recall  f1-score   support

             ADJ       0.85      0.76      0.80      1692
             ADP       0.93      0.94      0.94      2019
             ADV       0.89      0.84      0.87      1226
             AUX       0.96      0.97      0.96      1503
           CCONJ       0.99      0.99      0.99       739
             DET       0.98      0.98      0.98      1896
            INTJ       0.81      0.65      0.72       120
            NOUN       0.82      0.86      0.84      4133
             NUM       0.92      0.83      0.88       536
            PART       0.92      0.98      0.95       630
            PRON       0.98      0.97      0.98      2158
           PROPN       0.78      0.84      0.81      2075
           PUNCT       0.99      0.99      0.99      3106
           SCONJ       0.86      0.73      0.79       386
             SYM       0.90      0.70      0.79        92
            VERB       0.88      0.89      0.89      2647
               X       0.66      0.52      0.58       139

       micro avg       0.90      0.90      0.90     25097
       macro avg       0.89      0.85      0.87     25097
    weighted avg       0.90      0.90      0.90     25097

## Remarks
In the **feature engineering** part, two changes are made:
- More features are extracted for each token (one more token to the left, one more token to the right, and the token's lemma)
- More models are enabled for training (stochastic gradient descent and decision tree)

It's noted that using stochastics gradient descent/decision tree doesn't improve over the original support vector machine. However, by extracting more features, the accuracy increased.
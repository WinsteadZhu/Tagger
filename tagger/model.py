from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from .pre_process import load_dataset, token_to_features, prepare_data_for_training, vectorize
import numpy as np
import pickle
import spacy
import time
from sklearn.metrics import classification_report

nlp = spacy.load("en_core_web_sm")

# Fit the training data using support vector machine and report the corss-validation score (if preferred)
def SVM_fit_and_report(X, Y, cross_val = True, n_folds=5):

    svm = LinearSVC()

    if cross_val:
        print(f"Doing {n_folds}-fold cross-validation.")
        scores = cross_val_score(svm, X, Y, cv=n_folds)
        print(f"{n_folds}-fold cross-validation results over training set:\n")
        print("Fold\tScore".expandtabs(15))
        for i in range(n_folds):
            print(f"{i+1}\t{scores[i]:.3f}".expandtabs(15))
        print(f"Average\t{np.mean(scores):.3f}".expandtabs(15))

    print("Fitting model.")
    start_time = time.time()
    svm.fit(X, Y)
    end_time = time.time()
    print(f"Took {int(end_time - start_time)} seconds.")

    return svm

# Fit the training data using stochastic gradient descent and report the corss-validation score (if preferred)
def SGD_fit_and_report(X, Y, cross_val = True, n_folds=5):

    sgd = SGDClassifier()

    if cross_val:
        print(f"Doing {n_folds}-fold cross-validation.")
        scores = cross_val_score(sgd, X, Y, cv=n_folds)
        print(f"{n_folds}-fold cross-validation results over training set:\n")
        print("Fold\tScore".expandtabs(15))
        for i in range(n_folds):
            print(f"{i+1}\t{scores[i]:.3f}".expandtabs(15))
        print(f"Average\t{np.mean(scores):.3f}".expandtabs(15))

    print("Fitting model.")
    start_time = time.time()
    sgd.fit(X, Y)
    end_time = time.time()
    print(f"Took {int(end_time - start_time)} seconds.")

    return sgd

# Fit the training data using decision tree and report the corss-validation score (if preferred)
def DT_fit_and_report(X, Y, cross_val = True, n_folds=5):

    dt = DecisionTreeClassifier()

    if cross_val:
        print(f"Doing {n_folds}-fold cross-validation.")
        scores = cross_val_score(dt, X, Y, cv=n_folds)
        print(f"{n_folds}-fold cross-validation results over training set:\n")
        print("Fold\tScore".expandtabs(15))
        for i in range(n_folds):
            print(f"{i+1}\t{scores[i]:.3f}".expandtabs(15))
        print(f"Average\t{np.mean(scores):.3f}".expandtabs(15))

    print("Fitting model.")
    start_time = time.time()
    dt.fit(X, Y)
    end_time = time.time()
    print(f"Took {int(end_time - start_time)} seconds.")

    return dt

# Binarize the model and vectorizer, and save to disk
def save_model(model_and_vec, output_file):
    print(f"Saving model to {output_file}.")
    with open(output_file, "wb") as outfile:
        pickle.dump(model_and_vec, outfile)

# Load model from disk
def load_model(output_file):
    print(f"Loading model from {output_file}.")
    with open(output_file, "rb") as infile:
        model, vec = pickle.load(infile)

    return model, vec

# Tag a single sentence
def tag_sentence(sentence, model, vec):
    doc = nlp(sentence)
    tokenized_sent = [token.text for token in doc]
    featurized_sent = []
    for i, token in enumerate(tokenized_sent):
        featurized_sent.append(token_to_features(tokenized_sent, i))

    featurized_sent = vec.transform(featurized_sent)
    labels = model.predict(featurized_sent)
    tagged_sent = list(zip(tokenized_sent, labels))

    return tagged_sent

def print_tagged_sent(tagged_sent):
    for token in tagged_sent:
        print(f"{token[0]}\t{token[1]}".expandtabs(15))

# Tag an entire text file
def tag_file(text, model, vec, file_name):

    tagged_text = []
    nlp.max_length = len(text)
    processed_text = nlp(text)
    for sent in processed_text.sents:
        tokenized_sent = [token.text for token in sent]
        featurized_sent = []
        for i, token in enumerate(tokenized_sent):
            featurized_sent.append(token_to_features(tokenized_sent, i))
        featurized_sent = vec.transform(featurized_sent)
        labels = model.predict(featurized_sent)
        tagged_sent = list(zip(tokenized_sent, labels))

        tagged_text.append(tagged_sent)

    new_file_name = file_name + ".tag"
    with open(new_file_name, "wt") as outfile:
        for tagged_sent in tagged_text:
            for token in tagged_sent:
                outfile.write(f"{token[0]}\t{token[1]}\n".expandtabs(15))
            outfile.write("\n")

# Tag a file using the saved model, and compare the results to the gold tags
def evaluate(X, Y, model, vec):
    X = vec.transform(X)
    Y_pred = model.predict(X)
    print(classification_report(Y, Y_pred))
    


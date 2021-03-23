from sklearn.feature_extraction import DictVectorizer

# Load .txt file
def load_txt(file):
    X, Y = [], []
    with open(file, "r") as infile:
        sents = infile.read().split("\n\n")
        if sents[-1] == "":
            sents = sents[:-1]
        for sent in sents:
            words, tags = [], []
            lines = sent.split("\n")
            for line in lines:
                line = line.strip().split("\t")
                if len(line) != 2:
                    raise TabError("Tried to read .txt file, but did not find two columns.")
                else:
                    words.append(line[0])
                    tags.append(line[1])
            X.append(words)
            Y.append(tags)
    # Because .txt file doesn't contain lemmas, create a list of value'None'
    lemmas = [None] * len(Y)

    return X, Y, lemmas

# Load .conllu file
def load_conllu(file):
    X, Y = [], []
    # Create a list for storing lemmas from .conllu file
    lemmas = []
    with open(file, "r") as infile:
        sents = infile.read().split("\n\n")
        if sents[-1] == "":
            sents = sents[:-1]
        for sent in sents:
            words, tags = [], []
            lemma = []
            lines = sent.split("\n")
            for line in lines:
                if line.startswith("#"):
                    continue
                line = line.strip().split("\t")
                if len(line) != 10:
                    raise TabError("Tried to read .txt file, but did not find ten columns.")
                else:
                    words.append(line[1])
                    tags.append(line[3])
                    lemma.append(line[2])
            X.append(words)
            Y.append(tags)
            lemmas.append(lemma)

    return X, Y, lemmas

# Load .txt  or .conllu file
def load_dataset(file):
    if file.endswith(".conllu"):
        try:
            X, Y, lemmas = load_conllu(file)
            return X, Y, lemmas
        except TabError:
            print("Tried to read .txt file, but did not find ten columns.")
    else:
        try:
            X, Y, lemmas = load_txt(file)
            return X, Y, lemmas
        except TabError:
            print("Tried to read .txt file, but did not find two columns.")

# Create feature dictionary for the i-th token of sent
def token_to_features(sent, i, lemma=None):
    word = sent[i]

    # Features of i-th token
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.lemma': lemma
    }
    
    # Features of (i-1)-th token
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    
    # Features of (i-2)-th token
    if i > 1:
        word2 = sent[i-2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
        })

    # Features of (i+1)-th token
    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    
    # Features of (i+2)-th token
    if i < len(sent)-2:
        word2 = sent[i+2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
        })

    return features

# Takes in sentence and tag lists and collapses into one list of tokens and one list of tags
def prepare_data_for_training(X, Y, lemmas):
    X_out, Y_out = [], []
    
    for i, sent in enumerate(X):
        for j, token in enumerate(sent):
            features = token_to_features(sent, j, lemmas[i][j])
            X_out.append(features)
            Y_out.append(Y[i][j])

    return X_out, Y_out

# Transform list of dictionaries into a feature matrix
def vectorize(X):
    vec = DictVectorizer()
    X_out = vec.fit_transform(X)

    return X_out, vec

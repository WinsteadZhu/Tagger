from tagger import pre_process, model
import argparse
import yaml
import os

def run(args):
    mode = args.mode
    
    with open(args.config, "r") as yamlin:
        config = yaml.load(yamlin)
        
    if mode == "train":
        print(f"Training a model on {config['train_file']}.")
        X, Y, lemmas = pre_process.load_dataset(config["train_file"])
        X, Y = pre_process.prepare_data_for_training(X, Y, lemmas)
        X, vec = pre_process.vectorize(X)
    
        # Use support vector machine for training
        if args.model == "svm":
            svm = model.SVM_fit_and_report(X, Y, config["crossval"], config["n_folds"])
            model.save_model((svm, vec), config["model_file"])
        
        # Use stochastic gradient descent for training
        elif args.model == "sgd":
            sgd = model.SGD_fit_and_report(X, Y, config["crossval"], config["n_folds"])
            model.save_model((sgd, vec), config["model_file"])
        
        # Use decision tree for training
        elif args.model == "dt":
            dt = model.DT_fit_and_report(X, Y, config["crossval"], config["n_folds"])
            model.save_model((dt, vec), config["model_file"])
            
    elif mode == "tag":
        print(f"Tagging text using pretrained model: {config['train_file']}.")
        trained_model, vec = model.load_model(config["model_file"])
        # Tag an entire text file
        if os.path.isfile(args.text):
            text = open(args.text, "r").read()
            file_name = args.text
            tagged_file = model.tag_file(text, trained_model, vec, file_name)
            print("File saved to the same directory as the input file.")
        # Tag a single sentence
        else:
            tagged_sent = model.tag_sentence(args.text, trained_model, vec)
            model.print_tagged_sent(tagged_sent)
            
    elif mode == "eval":
        print(f"Evaluating model on {args.gold}.")
        X, Y, lemmas = pre_process.load_dataset(args.gold)
        X, Y  = pre_process.prepare_data_for_training(X, Y, lemmas)
        trained_model, vec = model.load_model(config["model_file"])
        model.evaluate(X, Y, trained_model, vec)
    
    else:
        print(f"{args.mode} is an incompatible mode. Must be either 'train', 'tag' or 'eval'.")

if __name__ == '__main__':


    PARSER = argparse.ArgumentParser(description=
                                     """
                                     A basic SVM-based POS-tagger.
                                     Accepts either .conllu or tab-delineated
                                     .txt files for training.
                                     """)

    PARSER.add_argument('--mode', metavar='M', type=str, help=
                        """
                        Specifies the tagger mode: {train, tag, eval}.
                        """)
    PARSER.add_argument('--text', metavar='T', type=str, help=
                        """
                        Tags a sentence string or an entire text file.
                        Can only be called if '--mode tag' is specified.
                        """)
    PARSER.add_argument('--config', metavar='C', type=str, help=
                        """
                        A config .yaml file that specifies the train data,
                        model output file, and number of folds for cross-validation.
                        """)
    
    PARSER.add_argument('--gold', metavar='G', type=str, help=
                        """
                        A gold-standard test (or dev) file in either .txt or .conllu format.
                        """)
                        
    PARSER.add_argument('--model', metavar='ML', type=str, help=
                        """
                        Speficies the training model: {svm, sgd, dt}.
                        """)

    ARGS = PARSER.parse_args()

    run(ARGS)

from utils.dataset import Dataset
from utils.baseline import Baseline
from utils.scorer import report_score
from utils.wordvec import Word2vec
from random import seed
import numpy as np

# I used ideas from: 
# http://www.aclweb.org/anthology/S16-1157
# http://m-mitchell.com/NAACL-2016/SemEval/pdf/SemEval136.pdf
# Uncomment lines 19, 25 and 58 and comment lines 24 and 57 to use parts of the
# training data

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

#    trainset = data.trainset[:int(len(data.trainset)*1/100)]
    print('Feature based models')
    baseline = Baseline(language)
    
    print('Training models')
    baseline.train(data.trainset)
#    baseline.train(trainset)

    print('Predicting labels')
    predictions = baseline.test(data.devset)

    predictions_int =[]
    for pred in predictions:
        pred_int = []
        for val in pred[1]:
            pred_int.append(int(val))
        predictions_int.append(pred_int)

    gold_labels = [sent['gold_label'] for sent in data.devset]
#    target_words = [sent['target_word'] for sent in data.devset]
        
    print('Calculating scores')
    for pred in predictions:
        print('Scores for' ,pred[0])
        report_score(gold_labels, pred[1])
    
    print('Scores for hard voting with all models')
    avg_pred_int = np.mean(np.array(predictions_int), axis = 0).tolist()
    avg_pred = [str(round(val)) for val in avg_pred_int]
    report_score(gold_labels, avg_pred)
    
#   Woed2vec based models
    
    print('Word2vec based models')
    print('Loading w2v')
    w2v = Word2vec(language)
    
    print('Training models')
    w2v.train(data.trainset)
#    w2v.train(trainset)
    
    print('Predicting labels')    
    predictions_w2v = w2v.test(data.devset)
    
    predictions_w2v_int =[]
    for pred in predictions_w2v:
        pred_int = []
        for val in pred[1]:
            pred_int.append(int(val))
        predictions_w2v_int.append(pred_int)
    
    print('Calculating scores')
    for pred in predictions_w2v:
        print('Scores for' ,pred[0])
        report_score(gold_labels, pred[1])
    
    print('Scores for hard voting with all models')
    avg_pred_w2v_int = np.mean(np.array(predictions_w2v_int), axis = 0).tolist()
    avg_pred_w2v = [str(round(val)) for val in avg_pred_w2v_int]
    report_score(gold_labels, avg_pred_w2v)
    
    for pred in predictions:
        pred_int = []
        for val in pred[1]:
            pred_int.append(int(val))
        predictions_w2v_int.append(pred_int)
    
    print('Scores for hard voting with both types of models')
    avg_pred_all_int = np.mean(np.array(predictions_w2v_int), axis = 0).tolist()
    avg_pred_all = [str(round(val)) for val in avg_pred_all_int]
    report_score(gold_labels, avg_pred_all)
    
#   This part with commented line 38, 112 and 113 were used to get examples of
#   target words that were correctly predicted by the improvement but not by baseline
#    k = 0
#    label_examples = []
#    label_examples_2 = []
#    label_examples_3 = []
#    for label in gold_labels:
#        if baseline_pred[k] != label and avg_pred_all[k] == label:
#            label_examples.append((target_words[k], label))
#        if baseline_pred[k] == label and avg_pred_all[k] != label:
#            label_examples_2.append((target_words[k], label))
#        if baseline_pred[k] != label and avg_pred_all[k] != label:
#            label_examples_3.append((target_words[k], label))
#        k += 1
#    
#    return label_examples, label_examples_2, label_examples_3
    
if __name__ == '__main__':
    seed(100)
    execute_demo('english')
    execute_demo('spanish')
#    words_eng1, words_eng2, words_eng3 = execute_demo('english')
#    words_spa1, words_spa2, words_spa3 = execute_demo('spanish')



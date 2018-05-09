import re, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from curses.ascii import isdigit
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from utils.syllable_spanish import silabizer
from langdetect import detect_langs

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        
        # http://wortschatz.uni-leipzig.de/en/download
        if language == 'english':
            self.avg_word_length = 5.3
            # from Wikipedia https://en.wikipedia.org/wiki/Letter_frequency
            self.lowfreqchar = re.compile('j|k|q|v|x|z')
            # for values with <1% realtive frecuancy (RF) value of 1 - RF
            self.lowfreqdict = {'j' : 0.847,'k' : 0.228, 'q': 0.905, 'v': 0.022,
                                'x': 0.85, 'z': 0.926}
            # from Wikipedia https://en.wikipedia.org/wiki/Letter_frequency
            self.highfreqchar = re.compile('e|t|a|o|i')
            # for 5 with higher realtive frecuancy (HF) value of HF/10
            self.highfreqdict = {'e' : 1.27,'t' : 0.906, 'a': 0.817, 'o': 0.751,
                                'i': 0.697}
            self.syll = cmudict.dict()
            self.vowels = re.compile('[aeiouyAEIOUY]')
            self.consonants = re.compile('[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]')
            self.mul_vow = re.compile('[aeiouyAEIOUY]{3,}')
            self.mul_cons = re.compile('[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM]{3,}')
            self.trans_cons = re.compile('[qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM](?![qwrtypsdfghjklzxcvbnmQWRTYPSDFGHJKLZXCVBNM])')
            self.trans_vow = re.compile('[aeiouyAEIOUY](?![aeiouyAEIOUY])')
            self.model_file = 'freq_datasets/word-freq-eng.pkl'
            nms = 1
            nm = 4
        else:  # spanish
            self.avg_word_length = 6.2
            # from Wikipedia https://en.wikipedia.org/wiki/Letter_frequency
            self.lowfreqchar = re.compile('f|h|j|k|q|w|x|z')
            # for values with <1% realtive frecuancy (RF) value of 1 - RF
            self.lowfreqdict = {'f' : 0.308,'h' : 0.297, 'j': 0.507, 'k': 0.989,
                                'q': 0.123, 'w': 0.983, 'x': 0.785, 'z': 0.533,
                                'ñ': 0.689}
             # from Wikipedia https://en.wikipedia.org/wiki/Letter_frequency
            self.highfreqchar = re.compile('e|a|o|s|r')
            # for 5 with higher realtive frecuancy (HF) value of HF/10
            self.highfreqdict = {'e' : 1.218,'a' : 1.153, 'o': 0.868, 's': 0.798,
                                'r': 0.687}
            self.syll = silabizer()
            self.vowels = re.compile('[aeiouAEIOUáéíóúÁÉÍÓÚ]')
            self.consonants = re.compile('[qwrtypsdfghjklñzxcvbnmQWRTYPSDFGHJKLÑZXCVBNM]')
            self.mul_cons = re.compile('[qwrtypsdfghjklñzxcvbnmQWRTYPSDFGHJKLÑZXCVBNM]{3,}')
            self.mul_vow = re.compile('[aeiouAEIOUáéíóúÁÉÍÓÚ]{3,}')
            self.accents = re.compile('[áéíúóÁÉÍÓÚ]')
            self.vowels2 = re.compile('[aeiouAEIOU]+')
            self.trans_cons = re.compile('[qwrtypsdfghjklñzxcvbnmQWRTYPSDFGHJKLÑZXCVBNM](?![qwrtypsdfghjklñzxcvbnmQWRTYPSDFGHJKLÑZXCVBNM])')
            self.trans_vow = re.compile('[aeiouAEIOUáéíóúÁÉÍÓÚ](?![aeiouAEIOUáéíóúÁÉÍÓÚ])')
            self.model_file = 'freq_datasets/word-freq-spa.pkl'
            nms = 5
            nm = 8
        with open(self.model_file, 'rb' ) as f:
            self.word_freq = pickle.load(f)
        self.doubleletter = re.compile(r'([a-zA-Z])\1')
#        nms = 0
#        nm = 10
        self.compiled = re.compile('\.|\,|\'|\"|\(|\)|«|»|’')
        self.models = [SVC(kernel="linear", C=0.025), KNeighborsClassifier(3),
                      SVC(gamma=2, C=1),
                      DecisionTreeClassifier(max_depth=5), 
                      RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                      AdaBoostClassifier(), LogisticRegression(), MLPClassifier(alpha=1),
                      GaussianNB(), QuadraticDiscriminantAnalysis()][nms:nm]
        self.names = ["Linear SVM", "Nearest Neighbors", "RBF SVM",
                      "Decision Tree", "Random Forest", "AdaBoost", "Logistic Regression",
                      "Neural Net", "Naive Bayes", "QDA"][nms:nm]
        print('Number of models:', len(self.models))
        

    def extract_features(self, words):
        len_chars = len(words) / self.avg_word_length
        len_tokens = len(words.split(' '))
        
        chars = re.findall(self.lowfreqchar, words)
        lf_sum = 1
        for char in chars:
            lf_sum += self.lowfreqdict[char]
        
        chars = re.findall(self.highfreqchar, words)
        hf_sum = 1
        for char in chars:
            hf_sum += self.highfreqdict[char]
            
        num_syll = 0
        for word in words.split():
            # Failed to get syllables from:
            # Turkish : معمار سينان
            try:
                num_syll += self.syllables(word)
            except:
                pass
        
        num_vowels = len(re.findall(self.vowels, words))
        
        num_consonants = len(re.findall(self.consonants, words))
        
        num_mul_vow = len(re.findall(self.mul_vow, words))
        
        num_mul_cons = len(re.findall(self.mul_cons, words))
        
        num_double_char = len(re.findall(self.doubleletter, words))
        
        num_vow_to_cons = re.findall(self.trans_cons, words)
        num_cons_to_vow = re.findall(self.trans_vow, words)
        num_total_trans = len(num_vow_to_cons) + len(num_cons_to_vow)
        
        words_list = words.lower().split()
        
        target_freq = 0
        
        for word in words_list:
            if word in self.word_freq:
                target_freq += self.word_freq[word]
            else:
                target_freq += 0.05
        
        senses = 0
        synonyms = 0
        hypernyms = 0
        hyponyms = 0
        len_def = 0
        num_POS = 0
        
        if self.language == 'spanish':
            for word in words_list:
                word_synset = wn.synsets(word, lang = 'spa')
                senses += len(word_synset)
                synonyms += sum([len(syn.lemmas()) for syn in word_synset])
                hypernyms += sum([len(syn.hypernyms()) for syn in word_synset])
                hyponyms += sum([len(syn.hyponyms()) for syn in word_synset])
                if senses != 0:
                    len_def += sum([len(syn.definition().split()) for syn in word_synset])/senses
                num_POS += len(set([syn.pos()for syn in word_synset]))
        else:
            for word in words_list:
                word_synset = wn.synsets(word)
                senses += len(word_synset)
                synonyms += sum([len(syn.lemmas()) for syn in word_synset])
                hypernyms += sum([len(syn.hypernyms()) for syn in word_synset])
                hyponyms += sum([len(syn.hyponyms()) for syn in word_synset])
                if senses != 0:
                    len_def += sum([len(syn.definition().split()) for syn in word_synset])/senses
                num_POS += len(set([syn.pos()for syn in word_synset]))
        if senses == 0:
            for word in words_list:
                try:
                    word_synset = wn.synsets(wn.morphy(word))
                    senses += len(word_synset) # senses
                    synonyms += sum([len(syn.lemmas()) for syn in word_synset]) # synonyms
                    hypernyms += sum([len(syn.hypernyms()) for syn in word_synset]) # hypernyms
                    hyponyms += sum([len(syn.hyponyms()) for syn in word_synset]) # hyponyms
                    if senses != 0:
                        len_def += sum([len(syn.definition().split()) for syn in word_synset])/senses
                    num_POS += len(set([syn.pos()for syn in word_synset]))
                except:
                    pass
                
#        Tarda mucho
#        langs, problang = self.langfeatures(words)
                
        features = [len_chars, len_tokens, lf_sum/len_chars, hf_sum/len_chars, 
                    num_syll/len_tokens, num_vowels/len_tokens,
                    num_consonants/len_tokens, num_mul_vow/len_tokens,
                    num_mul_cons/len_tokens, num_double_char/len_tokens,
                    num_total_trans/len_chars, target_freq/len_tokens, 
                    senses/len_tokens, synonyms/len_tokens, hypernyms/len_tokens, 
                    hyponyms/len_tokens, len_def/len_tokens, num_POS/len_tokens]
                
        return features

    def train(self, trainset):
        X = []
        y = []
        i = 0
        for sent in trainset:
            target = re.sub(self.compiled, '', sent['target_word'])
            X.append(self.extract_features(target))
            y.append(sent['gold_label'])
            
        i = 0
        for model in self.models:
            print('Training: ', self.names[i])
            model.fit(X, y)
            i += 1

    def test(self, testset):
        X = []
        M = []
        for sent in testset:
            target = re.sub(self.compiled, '', sent['target_word'])
            M.append(self.extract_features(target))
        
        i = 0
        for model in self.models:
            X.append((self.names[i], model.predict(M)))
            i += 1
        return X
    
    # from https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
    def syllables(self, word):
        if self.language == 'spanish':
            return len(self.syll(word))
        count = 0
        if word.lower() in self.syll:
            return [len([y for y in x if isdigit(y[-1])]) for x in self.syll[word.lower()]][0]
        vowels = 'aeiouy'
        word = word.lower().strip(".:;?!")
        if word[0] in vowels:
            count +=1
        for index in range(1,len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count +=1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and (word[-3] not in vowels):
            count+=1
        if count == 0:
            count +=1
        return count
    
    def langfeatures(self, word):
        langs = 0
        prob = 0
        for i in range(10):
            valor = detect_langs(word)
            langs += len(valor)
            prob += valor[0].prob
        langs /= (i+1)
        prob /= (i+1)
        return langs, prob
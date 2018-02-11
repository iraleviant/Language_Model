import os
import torch
import sys


class Dictionary(object):
    """Build word2idx and idx2word from Corpus(train/val/test)"""
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word

    def add_word(self, word):
        """Create/Update word2idx and idx2word"""
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """Corpus Tokenizer"""
    def __init__(self, input_files):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(input_files)) #train
        #self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        #self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, input_files):
        """Tokenizes a text file."""
        ifs = input_files.split(",") 
        for path in ifs:
            print "Reading  ", path
            assert os.path.exists(path)
            n_sent = 0
            # Add words to the dictionary
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    n_sent+=1
                    if n_sent % 10000 == 0: #n_sent divides in 10000 without remainder
                        print  str(round(float(n_sent)/1000, 0))+'K'+'\r', 
                        sys.stdout.flush()
                    # line to list of token + eos
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
    
            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word] #assign each token in the corpus its index in the dictionary
                        token += 1

        return ids
    
###################################################################        
#### Test For me ###################################################
#########################################################################
def main():
    ###############################################################################
    # Load data
    ###############################################################################
    #### replace files' location accordingly, first better check on test files
    #input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/webbase_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/wiki_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/billion_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news_2013_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news2012_phrase2.txt"
    #input_files=   "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/billion_word_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/webbase_all_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2013_clean,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2012_clean" #clean without 2 phrase
    #input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/example_after_2phrase.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new_test.txt" #test
    input_files=   "/corpus/clean_wiki_new.txt,/corpus/billion_word_clean.txt,/corpus/webbase_all_clean.txt,/corpus/news_2013_clean,/corpus/news_2012_clean" #clean without 2 phrase
    input_test=  "/corpus/example_after_2phrase.txt,/corpus/clean_wiki_new_test.txt"
   
    corpus=Corpus(input_files)

    
    print('I''m here')

#################################################################     
########################

if __name__ == '__main__':
    main()

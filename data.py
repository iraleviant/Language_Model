import os
import torch
import sys
import re

Min_Freq=100

class Dictionary(object):
    """Build word2idx and idx2word from Corpus(train/val/test)"""
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = ['<unk>'] # position(index): word
        self.word2idx['<unk>']=0
        #self.files={} #name_file:number of lines
        self.counts={} #word: its count in the corpus
    
    def add_word(self, word):
        """Create/Update word2idx and idx2word"""
        
        if word in self.counts:
            self.counts[word]+=1
        else:
            self.counts[word]=1
            
        if word not in self.word2idx and self.counts[word]>=Min_Freq:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        
        return 

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
            n_lines_per_file = 0
            # Add words to the dictionary
            with open(path, 'r') as f:
                for line in f:
                    n_lines_per_file+=1
                    if n_lines_per_file % 10000 == 0: #n_sent divides in 10000 without remainder
                        print  str(round(float(n_lines_per_file)/1000, 0))+'K'+'\r', 
                        sys.stdout.flush()
                    # line to list of token + eos
                    words=re.findall(r'\w+', line) + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)
            #self.dictionary.files[path]= n_lines_per_file
        return 
    
    
    
#===============================================================================
# ###################################################################        
# #### Test For me ###################################################
# #########################################################################
# def main():
#     ###############################################################################
#     # Load data
#     ###############################################################################
#     #### replace files' location accordingly, first better check on test files
#     #input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/webbase_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/wiki_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/billion_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news_2013_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news2012_phrase2.txt"
#     #input_files=   "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/billion_word_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/webbase_all_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2013_clean,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2012_clean" #clean without 2 phrase
#     input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new_test.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/example_after_2phrase.txt" #test
#     #input_files=   "/corpus/clean_wiki_new.txt,/corpus/billion_word_clean.txt,/corpus/webbase_all_clean.txt,/corpus/news_2013_clean,/corpus/news_2012_clean" #clean without 2 phrase
#     #input_files=  "/corpus/example_after_2phrase.txt,/corpus/clean_wiki_new_test.txt"
#     
#     corpus=Corpus(input_files)
#  
#      
#     print('I''m here')
#  
# #################################################################     
# ########################
#  
# if __name__ == '__main__':
#     main()
#===============================================================================

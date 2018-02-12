import os
import torch
import sys


class Dictionary(object):
    """Build word2idx and idx2word from Corpus(train/val/test)"""
    def __init__(self):
        self.word2idx = {} # word: index
        self.idx2word = [] # position(index): word
        #self.files={} #name_file:number of lines
        
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
            n_lines_per_file = 0
            # Add words to the dictionary
            with open(path, 'r') as f:
                for line in f:
                    n_lines_per_file+=1
                    if n_lines_per_file % 10000 == 0: #n_sent divides in 10000 without remainder
                        print  str(round(float(n_lines_per_file)/1000, 0))+'K'+'\r', 
                        sys.stdout.flush()
                    # line to list of token + eos
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)
            #self.dictionary.files[path]= n_lines_per_file

        return 

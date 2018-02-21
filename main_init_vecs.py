import argparse
import time
import math
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import data
import model
from itertools import islice
import os
import itertools
import re
import codecs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from scipy.stats import spearmanr
import scipy.sparse as ss
import torch.utils.data as data_utils
# Add ckp

DIC_Ready=False #True if you already have readily available dictionary


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./inputSimple', # './input'
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='',
                    help='model checkpoint to use')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,#1
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20,#1 
                    metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35,#3
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./output/model.pt', # /output
                    help='path to save the final model')
parser.add_argument('--debug', type=bool,  default=False, # /output
                    help='in debug mode words are printed to screen')
args = parser.parse_args()

if args.debug:
    # debug settings fits LSTM.pdf
    args.data = './inputSimple'  
    args.epochs = 1  
    args.batch_size = 2
    args.bptt = 35
    args.dropout = 0.2
    #args.cuda=True
else:
    args.data = './input'
    args.epochs = 1 #40
    args.batch_size = 40 #100
    args.bptt = 30    
    args.dropout = 0

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
#input_files=   "../corpus/clean_wiki_new.txt,../corpus/billion_word_clean.txt,../corpus/webbase_all_clean.txt,../corpus/news_2013_clean,../corpus/news_2012_clean" #clean without 2 phrase
#input_test=  "../corpus/example_after_2phrase.txt,../corpus/clean_wiki_new_test.txt"
#input_files=   "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/billion_word_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/webbase_all_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2013_clean,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2012_clean" #clean without 2 phrase
input_files= "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new_test.txt"

print('starting loading test data')

if DIC_Ready:
    objects = []
    #with (open('savedDictionaryTest', "rb")) as openfile:
    with (open('savedDictionaryAll150', "rb")) as openfile:
    #with (open('savedDictionaryALL', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    corpus=objects[0]
    print('corpus-dictionary read')
else:
    corpus = data.Corpus(input_files)
    with open('savedDictionaryTest', 'wb') as fp:
        pickle.dump(corpus, fp)
    print('corpus-dictionary saved')


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def tokenize_per_chunck(next_lines):
  
    ids = torch.LongTensor(900000) #torch.LongTensor(tokens)
    token_ind = 0
    tokens_list=[]
    for line in next_lines:
        if not line:
            #     print 'Ive reached the end'
            break
        words=re.findall(r'\w+', line) + ['<eos>']
        for word in words:
            try:
                ids[token_ind] = corpus.dictionary.word2idx[word]
                tokens_list.append(corpus.dictionary.word2idx[word])
                token_ind += 1
            except:
                ids[token_ind]=0 ##OOV (out of vocabulary word), corpus.dictionary.word2idx[<unk>]=0 
                token_ind += 1
    ids_new=ids[0:token_ind]
    return set(tokens_list), ids_new
    
eval_batch_size = 10
#train_data = batchify(corpus.train, args.batch_size)  #args.batch_size = 2
#val_data = batchify(corpus.valid, eval_batch_size)
#test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)

def build_init_weights_matrix():
    dic_file_order='/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/cws_dictionary_allpats_python_order.dat'
    fread=codecs.open(dic_file_order)
    cws_clean={}
    
    lines_f = fread.readlines()[1:]
    for line_g in lines_f:
        line_f=line_g.strip()
        line=line_f.split(" ")
        cws_clean[line[0]]=line[1]
    print "Finished reading content word dictionary its length is:", len(cws_clean)
    mat=ss.load_npz('/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/svd_reduced_mat_200_sym_ant_pats_beta1.npz')
    
    missing=[]
    new_mat=nn.Embedding(ntokens, args.emsize)
    #new_mat= numpy.zeros((ntokens, args.emsize)) #200 is the vectors dimension
    for w in corpus.dictionary.word2idx:
        row_new= corpus.dictionary.word2idx[w]
        if w in cws_clean:
            row_old=cws_clean[w]
            new_mat.weight.data[row_new]=torch.from_numpy(mat[row_old].toarray())
        else:
            #print "This word isn't in my vectors: ", w #if none initialize like they did with uniform uniform_(-0.1, 0.1)
            missing.append(w)
            X=numpy.random.uniform(low=-10, high=10, size=(args.emsize,)) #if none initialize like they did with uniform(-10,10)
            new_mat.weight.data[row_new]=torch.from_numpy(X)
    return new_mat

new_mat=build_init_weights_matrix()

model = model.RNNModel(args.model, ntokens, new_mat, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

# Load checkpoint
if args.checkpoint != '':
    if args.cuda:
        model = torch.load(args.checkpoint)
    else:
        # Load GPU model on CPU
        model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

if args.cuda:
    model.cuda()
else:
    model.cpu()
print (model)
#quit()

criterion = nn.CrossEntropyLoss()
if args.cuda:
    criterion.cuda()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target



def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def load_simL():
    fread_simlex=codecs.open("/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    #fread_simlex=codecs.open("english_simlex_word_pairs_human_scores.dat", 'r', 'utf-8')  #scores from felix hill
    words_dic={}
    pair_list_human=[]
    lines_f = fread_simlex.readlines()[1:]  # skips the first line of "number of vecs, dimension of each vec"
    for line_f in lines_f:
        tokens = line_f.split(",")
        word_i = tokens[0].lower()
        word_j = tokens[1].lower()
        score = float(tokens[2]) 
        if word_i in corpus.dictionary.word2idx  and word_j in corpus.dictionary.word2idx :
            pair_list_human.append(((corpus.dictionary.word2idx[word_i], corpus.dictionary.word2idx[word_j]), score)) 
        if word_i not in words_dic:
            if word_i in corpus.dictionary.word2idx:
                words_dic[word_i]=corpus.dictionary.word2idx[word_i]
            #else:
            #    words_dic[word_i]=0
        if word_j not in words_dic:
            if word_j in corpus.dictionary.word2idx:
                words_dic[word_j]=corpus.dictionary.word2idx[word_j]
            #else:
            #    words_dic[word_j]=0
    X_train, X_test= train_test_split(words_dic.values(), test_size=0.5, random_state=0)
    return set(X_train), pair_list_human


(words_set, pair_list_human)=load_simL()

def compute_spearman(pair_list_human):
    model_scores = {}
    for (x, y) in pair_list_human: #pair_list_human:((vanish,disappear),9.8)
        (word_i, word_j) = x
        v1=model.encoder.weight.data[word_i]
        v2=model.encoder.weight.data[word_j]
        #cosval= 1 - spatial.distance.cosine(v1, v2)
        cosval=cosine_similarity([v1.cpu().numpy()], [v2.cpu().numpy()])
        model_scores[(word_i, word_j)] = round(cosval,3)
     
    spearman_human_scores=[]
    spearman_model_scores=[]
         
    for position_1, (word_pair, score_1) in enumerate(pair_list_human):
        score_2 = model_scores[word_pair]
        spearman_human_scores.append(score_1)
        spearman_model_scores.append(score_2)  
     
    spearman_rho = spearmanr(spearman_human_scores, spearman_model_scores)
    #spearman_rho_test = spearmanr([1,2,3,4,5], [5,6,7,8,7])  # (corr=0.82, pval=0.088)    
    print "The spearman corr is: ",  round(spearman_rho[0], 3)
    
    return

def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    
    if args.debug:
        for colIdx in range(0, train_data.size(1)):
            wordsInColumn = []
            for rowIdx in range(0, train_data.size(0)):
                wordsInColumn.append(corpus.dictionary.idx2word[train_data[rowIdx,colIdx]])
            print('train_data column no. %d: %s' % (colIdx, wordsInColumn))
    #######################   new ########################################################
    ########################################################################################################
    ifs = input_files.split(",") 
    for path in ifs:
        print "Reading  ", path
        assert os.path.exists(path)
        
        count_100=0
        with open(path) as f:
            for next_lines in itertools.izip_longest(*[f]*100): #reads 100 lines at a time
                line_start_time = time.time()
                count_100+=1
                if count_100  % 1== 0:
                    compute_spearman(pair_list_human)
                #print "Another 100 lines were read:", count_100
                (tokens_set, ids)=tokenize_per_chunck(next_lines)
                
                #if not words_set.intersection(tokens_set):
                    #continue
                #else:
                train_data = batchify(ids, args.batch_size) 
                
                lineProcessingTimeMs = (time.time() - line_start_time)*1000
                print('Lines no. %d: data proccess time: %f ms' % (count_100, lineProcessingTimeMs))
                nnStartTime = time.time()
                
                for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):  #args.bptt = 3 
                    #print batch  
                    data, targets = get_batch(train_data, i)    
                           
                    if args.debug:
                        #for batchIdx in range(0 , args.batch_size):
                        for batchIdx in range(0 , int(data.data.shape[1])):
                            dataWordsInBatch = []
                            targetWordsInBatch = []             
                            #for wordIdx in range(0, args.bptt):
                            for wordIdx in range(0, int(data.data.shape[0])):
                                dataWordsInBatch.append(corpus.dictionary.idx2word[int(data.data.numpy()[wordIdx,batchIdx])])
                                targetWordsInBatch.append(corpus.dictionary.idx2word[int(targets.data.numpy()[batchIdx + wordIdx * args.batch_size])])                                
                            print('input %d to nn: data words in batch no. %d: %s' % (batch,batchIdx,dataWordsInBatch))            
                            print('input %d to nn: target words in batch no. %d: %s' % (batch,batchIdx,targetWordsInBatch))
                        
                            
                            
                    hidden = repackage_hidden(hidden)
                    model.zero_grad()
                    output, hidden = model(data, hidden)
                    
                    # understanding the model:
                    if (args.batch_size == 1 and args.dropout == 0):
                        import copy
                        hiddenSingle    = model.init_hidden(args.batch_size)
                        hiddenMultiple  = model.init_hidden(args.batch_size)
                        for dataIdx in range(0, len(data)):
                            output, hiddenSingle = model(data[dataIdx], hiddenSingle)
                        output, hiddenMultiple = model(data, hiddenMultiple)
                        # here I printed hiddenMultiple and hiddenSingle to screen and saw they are identical
                    
                    loss = criterion(output.view(-1, ntokens), targets)
                    loss.backward()
            
                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                    for p in model.parameters():  #len(p)=dictionary size
                        p.data.add_(-lr, p.grad.data)
            
                    total_loss += loss.data
            
                    if batch % args.log_interval == 0 and batch > 0:
                        cur_loss = total_loss[0] / args.log_interval
                        elapsed = time.time() - start_time
                        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, batch, len(train_data) // args.bptt, lr,
                            elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                        total_loss = 0
                        start_time = time.time()
                    
                netProcessingTimeMs = (time.time() - nnStartTime)*1000
                print('Lines no. %d: net train time: %f ms' % (count_100, netProcessingTimeMs))

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        
        train()
        '''
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.        
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
        '''
        lr /= 1.01
        with open("output", 'wb') as f:
            torch.save(model, f)
        with open('embeddings', 'wb') as fp:
            pickle.dump(model.encoder.weight.data, fp)
        print "I'm here"
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

quit()

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

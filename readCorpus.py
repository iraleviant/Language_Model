import data

#### replace files' location accordingly, first better check on test files

###############################################################################
# Load data
###############################################################################

#input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/webbase_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/wiki_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/billion_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news_2013_phrase2.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/word_2phrase_corpus/news2012_phrase2.txt"
#input_files=   "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/billion_word_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/webbase_all_clean.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2013_clean,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/news_2012_clean" #clean without 2 phrase
#input_files = "/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/example_after_2phrase.txt,/home/ira/Dropbox/IraTechnion/Patterns_Research/sp_sg/clean_corpus_english/clean_wiki_new_test.txt" #test
input_files=   "/corpus/clean_wiki_new.txt,/corpus/billion_word_clean.txt,/corpus/webbase_all_clean.txt,/corpus/news_2013_clean,/corpus/news_2012_clean" #clean without 2 phrase
input_test=  "/corpus/example_after_2phrase.txt,/corpus/clean_wiki_new_test.txt"
corpus=data.Corpus(input_files)

print('I''m here')

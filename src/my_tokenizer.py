import spacy
import MeCab
from const import *

def get_tokenizer(type_of_tokenizer):
    if type_of_tokenizer=='mecab':
        tagger = MeCab.Tagger(f'-Owakati -d {DIR_MECAB_DIC}')
        tokenizer = lambda text: tagger.parse(text).split(' ')[:-1]
    elif type_of_tokenizer=='sudachi':
        nlp = spacy.load('ja_ginza')
        tokenizer = lambda text: [str(token) for token in nlp(text)] # text to list of tokens

    return tokenizer

if __name__=='__main__':
    print('mecab  :', get_tokenizer('mecab')(SAMPLE_SENT))
    print('sudachi:', get_tokenizer('sudachi')(SAMPLE_SENT))
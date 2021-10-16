import MeCab
from tokenizer import Tokenizer
from const import *

class MecabTokenizer(Tokenizer):
    def __init__(self):
        tagger = MeCab.Tagger(f'-Owakati -d {DIR_MECAB_DIC}')
        tokenizer = lambda text: tagger.parse(text).split(' ')[:-1]
        super().__init__(tokenizer)
    def tokenized_text(self, text):
        for token in self.tokenizer(text):
            yield token
    def tokenized_corpus(self, corpus):
        for text in corpus:
            yield self.tokenized_text(text)

if __name__ == '__main__':
    tokenizer = MecabTokenizer()
    corpus = [
        '国家公務員の叛逆',
        'ヴェニスの商人を返してくださいよ'
    ]
    print('test: 文 -> Generator of トークン')
    text = tokenizer.tokenized_text(corpus[0])
    for token in text:
        print(token)
    print('test: Iterable of 文 -> Nested generator of トークン')
    for text in tokenizer.tokenized_corpus(corpus):
        for token in text:
            print(token)

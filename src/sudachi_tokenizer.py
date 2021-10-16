from sudachipy import tokenizer, dictionary
from tokenizer import Tokenizer

class SudachiTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(dictionary.Dictionary().create())
    def tokenized_text(self, text):
        for token in self.tokenizer.tokenize(text):
            yield token.surface()
    def tokenized_corpus(self, corpus):
        for text in corpus:
            yield self.tokenized_text(text)

if __name__ == '__main__':
    tokenizer = SudachiTokenizer()
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

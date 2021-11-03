import MeCab
class JapaneseTokenizer:
    def __init__(self):
        self.mecab = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        self.mecab.parseToNode('')
    def split(self, text: str):
        node = self.mecab.parseToNode(text)
        words = []
        while node:
            if node.surface:
                words.append(node.surface)
            node = node.next
        return words
def tokenize(text: str):
    tokenizer = JapaneseTokenizer()
    return tokenizer.split(text)

if __name__ == '__main__':
    tokenize('私の家に遊びにおいでよ')
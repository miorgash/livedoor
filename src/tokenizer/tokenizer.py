class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def tokenized_text(self):
        '''文字列を分かち書きする
        Args:
          text: 文字列
        Yields:
          トークン
        '''
        raise NotImplementedError
    def tokenized_corpus(self):
        '''コーパスを分かち書きする
        Args:
          corpus: 文字列のリスト
        Yields:
          テキストを分割したトークン列を返すジェネレータ
        '''
        raise NotImplementedError

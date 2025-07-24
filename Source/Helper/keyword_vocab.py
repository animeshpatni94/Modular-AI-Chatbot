from typing import Dict, List

class KeywordVocabulary:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.index = 0

    def get_index(self, keyword: str) -> int:
        if keyword not in self.vocab:
            self.vocab[keyword] = self.index
            self.index += 1
        return self.vocab[keyword]

    def build_from_keywords(self, keyword_lists: List[List[str]]):
        for keywords in keyword_lists:
            for kw in keywords:
                self.get_index(kw)

    def load_from_dict(self, vocab_dict: dict):
        self.vocab = vocab_dict
        self.index = len(vocab_dict)
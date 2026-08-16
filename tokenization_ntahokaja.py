from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer as RawTokenizer
import os
import torch

class NtahokajaTokenizer(PreTrainedTokenizer):
    """
    Nyehashi AI Sovereign Tokenizer.
    V0.31 - Production Grade. 
    Fixed: Missing self. reference in morphological engine.
    """
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file=None, **kwargs):
        if tokenizer_file is None:
            tokenizer_file = os.path.join(os.path.dirname(__file__), "tokenizer.json")
            
        self.backend_tokenizer = RawTokenizer.from_file(tokenizer_file)
        
        self.NOUN_PREFIXES = sorted(["umw", "umu", "um", "aba", "ab", "ba", "imi", "im", "iki", "ic", "ik", "ibi", "ivy", "ib", "inz", "iny", "in", "iz", "uru", "urw", "ur", "aka", "ag", "ak", "utu", "ut", "ubu", "ubw", "ub", "uku", "ukw", "aha", "ah"], key=len, reverse=True)
        self.SUBJ_PREFIXES = sorted(["nd", "tur", "mur", "bar", "nz", "uz", "az", "a", "u", "ba", "i", "ki", "ri", "ru", "ka", "bu", "ku", "ha", "tu", "mu", "n", "tw", "mw"], key=len, reverse=True)
        self.TENSE_MARKERS = sorted(["ra", "a", "ara", "zo", "ro", "o", "ki", "cha", "ka", "ri", "bwa"], key=len, reverse=True)
        self.DERIV_SUFFIXES = sorted(["ish", "esh", "ir", "er", "ik", "ek", "w", "an", "ur", "uk", "agur", "iriz", "eriz", "an", "gan", "rar", "rik", "rir", "rek", "ng", "ny", "z", "y", "ij", "ej", "iy", "ey"], key=len, reverse=True)
        self.COMMON_ROOTS = ["genda", "kora", "rim", "shing", "ntah", "vug", "meny", "bon", "gir", "shik", "banz", "bandany", "seng", "hamb", "tw", "ror", "tum", "vyar", "hamagar", "rek", "garuk", "teg", "tegerez", "baz", "shub", "shu", "shir", "shikir", "hish", "fuk", "hun", "hor", "her", "ran", "som", "sek", "ton", "ch", "ng", "nd"]
        
        super().__init__(**kwargs)

    def _segment_word(self, word):
        word = word.lower()
        if len(word) >= 2 and word[1] == "'": return word
        for p in self.NOUN_PREFIXES:
            if word.startswith(p) and len(word) > len(p):
                return p + "@@" + self._segment_word(word[len(p):])
        for sp in self.SUBJ_PREFIXES:
            if word.startswith(sp) and len(word) > len(sp):
                remainder = word[len(sp):]
                for tm in self.TENSE_MARKERS:
                    if remainder.startswith(tm):
                        after_tm = remainder[len(tm):]
                        for ds in self.DERIV_SUFFIXES:
                            if after_tm.endswith(ds) and len(after_tm) > len(ds):
                                root = after_tm[:-len(ds)]
                                if root in self.COMMON_ROOTS: return f"{sp}@@{tm}@@{root}@@{ds}"
                        if after_tm in self.COMMON_ROOTS: return f"{sp}@@{tm}@@{after_tm}"
                return sp + "@@" + remainder
        return word

    @property
    def vocab_size(self):
        return self.backend_tokenizer.get_vocab_size()

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()

    def _tokenize(self, text):
        prepped = " ".join([self._segment_word(w) for w in text.split()])
        return self.backend_tokenizer.encode(prepped, add_special_tokens=False).tokens

    def _convert_token_to_id(self, token):
        return self.backend_tokenizer.token_to_id(token) or self.backend_tokenizer.token_to_id("<unk>")

    def _convert_id_to_token(self, index):
        return self.backend_tokenizer.id_to_token(index)

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        text = self.backend_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text.replace("@@", "")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return tuple()

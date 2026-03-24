from collections.abc import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # Construct a tokenizer from a given
        # vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
        # the following parameters:
        # vocab: dict[int, bytes]
        # merges: list[tuple[bytes, bytes]]
        # special_tokens: list[str] | None = None
        self.vocab = vocab
        self.merges = merges 
        self.special_tokens = special_tokens

        self.tokens = {}
        if self.vocab:
            for i in self.vocab:
                # self.tokens.add(self.vocab[i])
                self.tokens[self.vocab[i]] = i

    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        # Class
        # method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        # (in the same format that your BPE training code output) and (optionally) a list of special
        # tokens. This method should accept the following additional parameters:
        # vocab_filepath: str
        # merges_filepath: str
        # special_tokens: list[str] | None = None
        with open(vocab_filepath, 'r') as fv:
            self.vocab = fv.read()
        with open(merges_filepath, 'r') as fm:
            self.merges = fm.read()
        self.special_tokens = special_tokens
            
        self.tokens = {}
        for i in self.vocab:
            # self.tokens.add(self.vocab[i])
            self.tokens[self.vocab[i]] = i

    def encode(self, text: str) -> list[int]:
        #  Encode an input text into a sequence of token IDs.
        # init
        ret = []
        # edge case: emptry str
        if len(text) == 0 or not text:
            return ret 
        
        i = 0
        while i < len(text):
            for j in range(i+1, len(text)+1): #+1 to allow j = len(text)
                if text[i:j] in self.tokens and j < len(text):
                    continue 
                elif j == i+1 and text[i:j] not in self.tokens: #if_0, not in self.token
                    self.tokens[text[i:j]] = len(self.tokens)
                    token = text[i:j]
                    i = j
                elif text[i:j] not in self.tokens: # if_1, not in self.token; 
                    token = text[i:j-1]
                    i = j-1
                elif text[i:j] in self.tokens and j == len(text):    # if_2, in self.tokens and end of text
                    token = text[i:j]
                    i = j
                else:
                    print('error, out of condition')
                break
            token_encoded = self.tokens[token]
            ret.append(token_encoded)

        return ret 
    


# # from tokenizer import Tokenizer
# # t = tokenizer({1:'a'}, [['1', 'a']])
# # t.encode('aaa ')
# # import tokenizer_cls
# # import importlib 
# # importlib.reload(tokenizer_cls)
# text = 'aaab'
# print(text)
# ret = []
# if len(text) == 0 or not text:
#     print(ret)

# i = 0
# while i < len(text):
#     for j in range(i+1, len(text)+1): 
#         if text[i:j] in t.tokens and j < len(text):
#             continue 
#         elif j == i+1 and text[i:j] not in t.tokens: #if_0, not in token
#             t.tokens[text[i:j]] = len(t.tokens)
#             token = text[i:j]
#             i = j
#         elif text[i:j] not in t.tokens: # if_1, not in token; 
#             token = text[i:j-1]
#             i = j-1
#         elif text[i:j] in t.tokens and j == len(text):    # if_2, in t.tokens and end of text
#             token = text[i:j]
#             i = j
#         else:
#             print('error, out of condition')
#         break
#     print('token, t.token', token, t.tokens, i, j, ret)
#     token_encoded = t.tokens[token]
#     ret.append(token_encoded)
#     print('here, i, j, ret', i, j, ret)
#     time.sleep(1)
    
# ret 

# import time 


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        #  Given an iterable of
        # strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        # required for memory-efficient tokenization of large files that we cannot directly load into
        # memory.

        ret = []

        with open(iterable, 'r') as f:
                # Move f to the start
                start = 0
                f.seek(start)
                len = 0
                # Read from f
                while f.seek(start):
                    #TODO if len(new_token) == 1, but not in self.tokens (likely impossible)
                    len += 1
                    new_token = f.read(len)
                    if new_token in self.tokens:
                        continue
                    else:
                        last_token = f.read(len-1)
                        last_token_encoded = self.tokens(last_token)
                        ret.append(last_token_encoded)

        return ret 

    def decode(self, ids: list[int]) -> str:
        #  Decode a sequence of token IDs into text.
        ret = []
        for i in ids:
            ret.append(self.vocab[i])
        
        return ''.join(ret)


    # To test your Tokenizer against our provided tests, you will first need to implement the test adapter
    # at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your implementation should be able to pass all tests.
            
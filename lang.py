import re
import unicodedata

MAX_LENGTH = 10

question_words = (
    'who', 'whom', 'what', 'which', 'whose',
    'where', 'when', 'why', 'how',
    'am', 'is', 'are', 'was', 'were',
    'do', 'does', 'did', 'have', 'has',
    'can', 'could', 'will', 'would',
    'shall', 'should', 'may', 'might'
)

good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)

class Lang:
    def __init__(self): 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "<NUMBER>"}
        self.n_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c)!= 'Mn'
    )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def get_pairs(input_file):    
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(string) for string in line.split('\t')] for line in lines]

    return pairs

def filter_pair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1].split(' ')) < MAX_LENGTH and pair[0].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(input_file):
    input_lang = Lang()
    output_lang = Lang()

    pairs = get_pairs(input_file)
    initial_length = len(pairs)

    pairs = filter_pairs(pairs)
    # print('filtered from %s to %s pairs' % (initial_length, len(pairs)))

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    return input_lang, output_lang, pairs

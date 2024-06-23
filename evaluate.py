import torch
from torch import optim
import random
import re

from model import EncoderRNN, AttnDecoderRNN
from lang import unicode_to_ascii, MAX_LENGTH

from train import learning_rate, hidden_size, n_layers, dropout_p

model_parts = torch.load('models/models.pth')

input_lang = model_parts['input_lang']
output_lang = model_parts['output_lang']
pairs = model_parts['pairs']

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

encoder.load_state_dict(model_parts['encoder'])
decoder.load_state_dict(model_parts['decoder'])
encoder_optimizer.load_state_dict(model_parts['encoder_optimizer'])
decoder_optimizer.load_state_dict(model_parts['decoder_optimizer'])

SOS_token = 0
EOS_token = 1

# def levenshtein_distance(a, b):
#     n, m = len(a), len(b)
#     if n > m:
#         a, b = b, a
#         n, m = m, n

#     current_row = range(n + 1)
#     for i in range(1, m + 1):
#         previous_row, current_row = current_row, [i] + [0] * n
#         for j in range(1, n + 1):
#             add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
#             if a[j - 1] != b[i - 1]:
#                 change += 1
#             current_row[j] = min(add, delete, change)

#     return current_row[n]

# def closest_word(word, vocab):

def indexes_from_sentence(lang, sentence):
    # need to find a way to get a word that's not in the vocab
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = torch.tensor(indexes).view(-1, 1)
    return var

def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

def evaluate(sentence, max_length = MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_context = torch.zeros(1, decoder.hidden_size)

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()

        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = torch.LongTensor([[ni]])
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly():
    pair = random.choice(pairs)
    
    output_words, _ = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')

def evaluate_input():
    while True:
        message = input("> ")

        message = unicode_to_ascii(message.lower().strip())
        message = re.sub(r"([.!?])", r" \1", message)
        message = re.sub(r"[^a-zA-Z!?]+", r" ", message)

        output_words, _ = evaluate(message)
        output_sentence = ' '.join(output_words[:-1])

        print(">", output_sentence)

if __name__ == '__main__':
    evaluate_input()
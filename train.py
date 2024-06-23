import time
import random
import math
import torch
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lang import prepare_data
from model import EncoderRNN, AttnDecoderRNN

input_file = 'data/eng-fra.txt'
input_lang, output_lang, pairs = prepare_data(input_file)

SOS_token = 0
EOS_token = 1
number_token = 2

teacher_forcing_ratio = 0.5
clip = 5.0

hidden_size = 500
n_layers = 2
dropout_p = 0.05

n_epochs = 50000
learning_rate = 0.0001

def indexes_from_sentence(lang, sentence):
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

def train_epoch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_context = torch.zeros(1, decoder.hidden_size)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, _ = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, _ = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = torch.LongTensor([[ni]])

            if ni == EOS_token: break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    # return '%s (Estimated Remaining: %s)' % (asMinutes(s), asMinutes(rs))
    return '(Estimated Remaining: %s)' % asMinutes(rs)

def show_plot(points):
    plt.plot(points)
    plt.show()

def train(n_epochs, learning_rate):
    print("Training the model...")
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    plot_every = n_epochs / 10
    print_every = n_epochs / 10

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    for epoch in range(1, n_epochs+1):
        training_pair = variables_from_pair(random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train_epoch(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0 or epoch == 1:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%d%% %s Epoch: %d, Loss: %.4f' % (epoch / n_epochs * 100, timeSince(start, epoch / n_epochs), epoch, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'input_lang': input_lang,
        'output_lang': output_lang,
        'pairs': pairs,
    }, 'models/models.pth')

    show_plot(plot_losses)

if __name__ == '__main__':
    train(n_epochs, learning_rate)
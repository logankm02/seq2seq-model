import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers=1):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.input_size = input_size
            self.hidden_size = hidden_size

            self.embedding = nn.Embedding(input_size, hidden_size)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

        def forward(self, word_inputs, hidden):
            seq_len = len(word_inputs)
            embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
            output, hidden = self.gru(embedded, hidden)
            return output, hidden
        
        def init_hidden(self):
            hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
            return hidden
        
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)


    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        attn_energies = Variable(torch.zeros(seq_len))

        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        hidden = hidden.squeeze(0)
        encoder_output = encoder_output.squeeze(0)
        energy = self.attn(encoder_output)
        energy = hidden.dot(energy)
        return energy
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers, dropout_p):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        self.attn = Attn(hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        word_embedded = self.embedding(word_input).view(1, 1, -1)

        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)

        return output, context, hidden, attn_weights
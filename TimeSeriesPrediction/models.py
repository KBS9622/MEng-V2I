import torch
import torch.nn as nn

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Vanilla RNN #
class RNN(nn.Module):
    """Recurrent Neural Network"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        out, h_out = self.rnn(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out


# Vanilla LSTM #
class LSTM(nn.Module):
    """Long Short Term Memory"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


# Vanill GRU #
class GRU(nn.Module):
    """Gate Recurrent Unit"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).float().to(device)

        _, h_out = self.gru(x, h_0)
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out
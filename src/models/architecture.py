import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    Arquitectura Recurrente LSTM Unidireccional.
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    """
    Arquitectura Recurrente GRU (Gated Recurrent Unit - 2 Compuertas).
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.4):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    """
    Arquitectura LSTM Bidireccional (BiLSTM).
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.4):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # Combinar el último paso temporal de la dirección forward y el primero de backward
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class TemporalAttention(nn.Module):
    """
    Mecanismo de Atención Temporal Bahdanau sobre secuencias RNN/LSTM.
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attn(lstm_output), dim=1) # (batch, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1) # (batch, hidden_size)
        return context_vector, attn_weights


class AttentionLSTMModel(nn.Module):
    """
    Arquitectura LSTM con Mecanismo de Atención Temporal.
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, dropout=0.4):
        super(AttentionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        context = self.dropout(context)
        out = self.fc(context)
        return out

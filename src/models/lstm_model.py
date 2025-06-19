# lstm_model.py
import torch
import torch.nn as nn

class PitchClassifierLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=1, num_classes=3):
        super(PitchClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]         # Take the last time step
        logits = self.classifier(final_output)    # (batch_size, num_classes)
        return logits
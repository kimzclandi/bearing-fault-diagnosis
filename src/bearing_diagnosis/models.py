import torch
from torch import nn

class BearingCNN(nn.Module):
    """[B,1,L] -> [B,4]; strided blocks keep CPU execution small."""
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,16,31,stride=8,padding=15),nn.BatchNorm1d(16),nn.ReLU(),
            nn.Conv1d(16,32,9,stride=4,padding=4),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,7,stride=4,padding=3),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.15),nn.Linear(64,4))
    def forward(self,x): return self.head(self.features(x))

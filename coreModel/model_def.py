import torch
import torch.nn as nn

class ResNetClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super(ResNetClassifier, self).__init__()
        self.encoder = encoder
        # Tête de classification personnalisée pour le diagnostic médical
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

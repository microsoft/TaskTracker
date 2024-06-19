import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

class BaseLinearProcessingModel(nn.Module):
    def __init__(self, num_layers: Union[int, Tuple[int, int]], feature_dim : int = 256, dropout: float = 0.5, conv=False, layer_norm=False, pool_first_layer=3):
        super().__init__()
        
        self.num_layers = self._get_num_layers(num_layers)
        
        # Each layer's final feature dimension
        self.feature_dim = feature_dim
        
        self.pool_first_layer = pool_first_layer
        
        # Define a series of layers for each layer of activation
        self.dropout = nn.Dropout(dropout)
        self.layers_fc = nn.ModuleList([self._create_fc_sequence() for _ in range(self.num_layers)])
        
        
        # layer to process concatenated features from all layers
        self.final_fc = nn.Linear(self.num_layers * feature_dim, 1024)
        self.conv = conv 
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.num_layers * feature_dim)

    def _get_num_layers(self, layers: Union[int, Tuple[int, int]]):
        if isinstance(layers, int):
            return layers
        elif isinstance(layers, tuple):
            start_idx, end_idx = layers[0], layers[1]
            return end_idx - start_idx + 1
        else:
            raise ValueError(f'Unsupported input layer type {layers}')
        
    def _create_fc_sequence(self):
        # This method should be implemented in derived classes
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, x):
        # x shape expected: [batch_size, num_layers, 4096] for last-token activations
        # Process each layer's activation through its FC sequence and store the result
        if self.conv:
            processed_layers = [fc(x[:, i, :].unsqueeze(1)) for i, fc in enumerate(self.layers_fc)]
        else:
            processed_layers = [fc(x[:, i, :]) for i, fc in enumerate(self.layers_fc)]
        
        # Concatenate the processed layers: [batch_size, num_layers * feature_dim]
        x = torch.cat(processed_layers, dim=1)

        if hasattr(self, 'layer_norm'):
            x = self.layer_norm(x)

        # Pass the concatenated features through the final dropout and FC layer
        x = self.final_fc(x)

        # Normalize the output embeddings
        return F.normalize(x, p=2, dim=-1)

class ParallelLinearProcessingModel(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's activations
        fc_sequence = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            self.dropout,
            nn.Linear(2048, 1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1024, 512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.ReLU(),
            self.dropout
        )
        return fc_sequence
    
class ParallelLinearProcessingModelWithLayerNorm(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's 
        # activations with Layer Normalization
        fc_sequence = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            self.dropout,
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            self.dropout,
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            self.dropout,
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            self.dropout
        )
        return fc_sequence

class ParallelLinearProcessingModelCompact(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's activations
        fc_sequence = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout
        )
        return fc_sequence

class ParallelLinearProcessingModelCompactV2(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's activations
        fc_sequence = nn.Sequential(
            nn.Linear(4096, 200),
            nn.ReLU(),
            self.dropout
        )
        return fc_sequence    

class ParallelConvProcessingModel(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's activations
        fc_sequence = nn.Sequential(
            nn.Conv1d(1, 7, kernel_size=70, stride=1),
            nn.ReLU(),          
            nn.MaxPool1d(kernel_size=self.pool_first_layer),
            self.dropout,  

            nn.Conv1d(7, 10, kernel_size=50, stride=1),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=3),
            self.dropout,  

            nn.Conv1d(10, 15, kernel_size=30, stride=1),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=3),
            self.dropout, 

            nn.Conv1d(15, 20, kernel_size=20, stride=1),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3),
            self.dropout,

            nn.Conv1d(20, 25, kernel_size=5, stride=1),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3),
            self.dropout,
            nn.Flatten()
        )
        return fc_sequence



class ParallelConvProcessingModelWide(BaseLinearProcessingModel):
    def _create_fc_sequence(self):
        # Creating a sequence of fully connected layers for each layer's activations
        
        fc_sequence = nn.Sequential(
            nn.Conv1d(1, 7, kernel_size=70, stride=1),
            nn.ReLU(),          
            nn.MaxPool1d(kernel_size=5),
            self.dropout,  

            nn.Conv1d(7, 10, kernel_size=50, stride=1),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=3),
            self.dropout,  

            nn.Conv1d(10, 15, kernel_size=30, stride=1),
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=3),
            self.dropout, 

            nn.Conv1d(15, 20, kernel_size=20, stride=1),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3),
            self.dropout,

            nn.Conv1d(20, 25, kernel_size=5, stride=1),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=3),
            self.dropout,
            nn.Flatten()
        )
        
        return fc_sequence

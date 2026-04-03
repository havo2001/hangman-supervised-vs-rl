from collections import namedtuple, deque

import torch
import random

import torch.nn as nn
import torch.nn.functional as F


State = namedtuple('State', ['word', 'guessed_char', 'remaining_guesses'])
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

# class DQN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(input_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, output_size)
    
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)


class DQN(nn.Module):
    def __init__(
        self,
        vocab_size=28, # 0-27
        embed_dim=64,
        lstm_hidden=128,
        lstm_layers=1,
        num_actions=26 # 26 letters
    ):
        super(DQN, self).__init__()

        # word encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=27)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        #  bidirectional
        lstm_output_dim = lstm_hidden * 2

        # concatenating:
        # guessed_letters: 26
        # remaining_guesses: 1
        fusion_input_dim = lstm_output_dim + 26 + 1

        self.fc1 = nn.Linear(fusion_input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Q-value head
        self.q_head = nn.Linear(128, num_actions)


    def forward(self, word_state, guessed_letters, remaining_guesses):
        """
        word_state:        (batch, 40)       LongTensor
        guessed_letters:   (batch, 26)       FloatTensor
        remaining_guesses: (batch, 1)        FloatTensor
        """
        if word_state.dim() == 1:
            word_state = word_state.unsqueeze(0) # (1, 40)
        if guessed_letters.dim() == 1:
            guessed_letters = guessed_letters.unsqueeze(0)   # (1, NUM_ACTIONS)
        if remaining_guesses.dim() == 1:
            remaining_guesses = remaining_guesses.unsqueeze(0)  #
        

        x = self.embedding(word_state) # (batch, 40, embed_dim)

        lstm_out, _ = self.lstm(x) # (batch, 40, 2*lstm_hidden)

        # Pooling 
        word_repr = torch.mean(lstm_out, dim=1)  # (batch, 2*lstm_hidden)

        # concatenate
        fused = torch.cat(
            [word_repr, guessed_letters, remaining_guesses],
            dim=1
        )

        # ffn
        x = F.relu(self.fc1(fused))
        x = F.relu(self.fc2(x))

        # ffn
        q_values = self.q_head(x) # (batch, 26)

        return q_values
    

        



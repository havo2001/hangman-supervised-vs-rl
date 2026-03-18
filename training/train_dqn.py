import argparse
import os
import math
from itertools import count
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from model.model_dqn import *

# Here / for padding, * for missing character
ALPHABET = 'abcdefghijklmnopqrstuvwxyz*/'
CHAR2ID  = {c:i for i,c in enumerate(ALPHABET)}
ID2CHAR  = {i:c for i,c in enumerate(ALPHABET)}
UNK_ID, PAD_ID = 26, 27
MAX_LEN = 40

# Input size of the DQN: length of the word + one-hot encoding for guessed character + remaining guesses
# 40 + 26 + 1 = 67
# Output size: number of actions = 26 (guessing each letter)
INPUT_SIZE = 67
NUM_ACTIONS = OUTPUT_SIZE = 26


def step(state: State, action: int, correct_word: str):
    "Take an action and return the next state and reward, whether the episode is done"
    new_word = ""
    guessed_char = state.guessed_char
    guessed_char.add(ID2CHAR[action])
    reward = -1

    for i, ch in enumerate(correct_word):
        # if we guessed the correct character, reveal it in the new word
        if ID2CHAR[action] == ch:
            reward = 1
            new_word += ch
        else:
            new_word += state.word[i]
    
    if reward == -1:
        remaining_guesses = state.remaining_guesses - 1
    else:
        remaining_guesses = state.remaining_guesses
    
    next_state = State(word=new_word, guessed_char=guessed_char, remaining_guesses=remaining_guesses)
    terminated = (new_word == correct_word) or (remaining_guesses <= 0)
    
    if new_word == correct_word:
        reward = 10 # big reward for winning
    elif remaining_guesses <= 0:
        reward = -10 # big penalty for losing
    
    return next_state, reward, terminated


def phi(state: State, device):
    "Convert the raw state into a feature vector for the DQN"
    word = [CHAR2ID.get(c, UNK_ID) for c in state.word.ljust(MAX_LEN, '/')]
    guessed_char = [1 if ID2CHAR[i] in state.guessed_char else 0 for i in range(NUM_ACTIONS)]
    remaining_guesses = state.remaining_guesses
    
    return torch.tensor(word + guessed_char + [remaining_guesses], dtype=torch.float32).to(device)


def select_action(policy_net, state, steps_done, eps_start, eps_end, eps_decay, device) -> int:
    # epsilon-greedy action selection
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    random_sample = random.random()
    
    if random_sample < eps_threshold:
        while True:    
            action = random.randint(0, NUM_ACTIONS - 1)
            if ID2CHAR[action] not in state.guessed_char:
                return action
    else:
        with torch.no_grad():
            q_values = policy_net(phi(state, device).unsqueeze(0))
            # Mask out already guessed characters by setting their Q-values to -inf
            for guessed in state.guessed_char:
                q_values[0, CHAR2ID[guessed]] = float('-inf')
            return q_values.argmax(dim=1).item()
        

def optimize_model(optimizer, policy_net, target_net, memory, batch_size, gamma, device):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    # https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    batch = Transition(*zip(*transitions)) #Check the documentation from pytorch for more details

    # compute non final mask
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                  device=device, dtype=torch.bool)
    
    non_final_next_states_list = [phi(s, device) for s in batch.next_state if s is not None]
    if non_final_next_states_list:
        non_final_next_states = torch.stack(non_final_next_states_list).to(device)
    else:
        non_final_next_states = None
    
    # Get the phi value of state_batch
    state_batch = torch.stack([phi(s, device) for s in batch.state]).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train_dqn(args):
    # Load the word list
    with open('data/train_data.txt', 'r') as f:
        word_list = [line.strip() for line in f]

    # They start identical
    policy_net = DQN(INPUT_SIZE, OUTPUT_SIZE).to(args.device)
    target_net = DQN(INPUT_SIZE, OUTPUT_SIZE).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())

    opt = optim.Adam(policy_net.parameters(), lr=args.lr, amsgrad=True)
    memory = ReplayMemory(args.replay_capacity)

    for episode in trange(args.num_episodes, desc="Training DQN"):
        word = word_list[random.randint(0, len(word_list) - 1)]
        state = State(word='*' * len(word), guessed_char=set(), remaining_guesses=6)
        for t in count():
            action = select_action(policy_net, state, episode, args.eps_start, args.eps_end, args.eps_decay, args.device)
            next_state, reward, terminated = step(state, action, word)

            if terminated:
                next_state = None
            
            memory.push(state, action, reward, next_state)
            state = next_state

            optimize_model(opt, policy_net, target_net, memory, args.batch_size, args.gamma, args.device)

            # soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)
            target_net.load_state_dict(target_net_state_dict)

            if terminated:
                break
        
    # Save the trained model
    # Create the folder if not exists yet
    os.makedirs('model/checkpoint', exist_ok=True)

    checkpoint = f'model/checkpoint/dqn_checkpoint.pt'
    torch.save(policy_net.state_dict(), checkpoint)
    print(f'Model checkpoint saved to {checkpoint}')

    print('Training complete!')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay_capacity', default=10000, type=int)
    ap.add_argument('--batch_size', default=128, type=int)
    ap.add_argument('--device', default='cuda', type=str)
    ap.add_argument('--num_episodes', default=10000, type=int)
    ap.add_argument('--gamma', default=0.99, type=float)
    ap.add_argument('--eps_start', default=0.9, type=float)
    ap.add_argument('--eps_end', default=0.01, type=float)
    ap.add_argument('--eps_decay', default=2500, type=int)
    ap.add_argument('--tau', default=0.005, type=float)
    ap.add_argument('--lr', default=3e-4, type=float) 
    args = ap.parse_args()

    train_dqn(args)
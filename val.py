import torch 
import numpy as np
from tqdm import tqdm
import argparse
import os

from utils.cnn_transformers_create_training_examples import *
from model.model_dqn import *
from training.train_dqn import *
from model.model_cnn_transformers import build_model


def simulate_test_game(word, model, model_type, device, max_wrong_guesses=6, verbose=2):
    '''
    Play a game of hangman with the given word using the model.
    Inputs:
        word: the word to guess
        model: the model to use for guessing
        model_type: 'supervise' or 'dqn'
        max_wrong_guesses: the maximum number of wrong guesses allowed
    Returns:
        True if the word was guessed correctly, False otherwise
    '''
    word_ids = {}
    for i, c in enumerate(word):
        if c not in word_ids:
            word_ids[c] = []
        word_ids[c].append(i)

    guessed_chars = set()
    encoded_word = '*' * len(word)
    num_guesses = 0


    while encoded_word != word and num_guesses < max_wrong_guesses:
        if verbose > 0:
            print(f'Current word: {encoded_word}')
            print(f'Guesses so far: {guessed_chars}')
        
        if model_type == 'supervise':
            final_chr, final_prob = None, None
            for i, c in enumerate(encoded_word):
                copy_word = encoded_word
                if c == '*':
                    copy_word = list(encoded_word)
                    copy_word[i] = MASK_CHAR
                    copy_word = ''.join(copy_word)

                    masked_info  = create_single_masked_word(copy_word, i)
                    if not masked_info:
                        continue

                    tok, _, idx, pad_mask = masked_info
                    tokens_tensor = torch.tensor(tok).unsqueeze(0).to(device)
                    mask_idx_tensor = torch.tensor(idx).to(device)
                    pad_mask_tensor = torch.tensor(pad_mask).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = model(tokens_tensor, mask_idx_tensor, pad_mask_tensor)
                        pred_idx = logits.argmax(dim=1).item()
                        predicted_char = ID2CHAR[pred_idx]

                        while predicted_char in guessed_chars:
                            logits[0, pred_idx] = -float('inf')
                            pred_idx = logits.argmax(dim=1).item()
                            predicted_char = ID2CHAR[pred_idx]

                        if final_chr is None or logits[0, pred_idx].item() > final_prob:
                            final_chr = predicted_char
                            final_prob = logits[0, pred_idx].item()
        else:
            state = State(word=encoded_word, guessed_char=guessed_chars, remaining_guesses=max_wrong_guesses-num_guesses)
            with torch.no_grad():
                q_values = model(*phi(state, device))
                for guessed in guessed_chars:
                    q_values[0, CHAR2ID[guessed]] = float('-inf')
                final_idx = q_values.argmax(dim=1).item()
                final_chr = ID2CHAR[final_idx]

        if final_chr in word:
            for idx in word_ids[final_chr]:
                encoded_word = encoded_word[:idx] + final_chr + encoded_word[idx + 1:]
        else:
            num_guesses += 1
        guessed_chars.add(final_chr)

        if verbose > 1:
            print(f'Guessing character: {final_chr}')
            print(f'Hangman state:', encoded_word)
            print(f'Number of wrong guesses: {num_guesses}')
    if verbose > 0:
        if encoded_word == word:
            print(f'Word guessed correctly: {word}')
            print('You win!')
        else:
            print(f'Word not guessed: {word}')
            print('You lose!')
    return encoded_word == word


def eval_subset(model, model_name, device, test_words):
    total_correct = 0
    test_words = [word for word in test_words if len(word) > 2] # Filter out words with length <= 2
    for word in tqdm(test_words):
        if simulate_test_game(word, model, model_name, device, max_wrong_guesses=6, verbose=0):
            total_correct += 1
    print(f'Total correct guesses: {total_correct}')
    print(f'Accuracy: {total_correct / len(test_words) * 100:.2f}%')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_type', default='dqn', type=str, choices=['supervise', 'dqn'])
    ap.add_argument('--model_checkpoint', default=1, type=str)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    ap.add_argument('--single_test_word', default=None, type=str)
    ap.add_argument('--test_file', default=None, type=str)

    args = ap.parse_args()

    if args.model_type == 'supervise':
        model = build_model().to(args.device)
        checkpoint = f'model/checkpoint/cnn_transformers_checkpoint_epoch_{args.model_checkpoint}.pt'
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
        print(f'Loaded model checkpoint from {checkpoint}')
    else:
        model = DQN().to(args.device)
        checkpoint = f'model/checkpoint/dqn_checkpoint_{args.model_checkpoint}.pt'
        model.load_state_dict(torch.load(checkpoint, map_location=args.device))
        print(f'Loaded model checkpoint from {checkpoint}')

    if args.single_test_word:
        simulate_test_game(args.single_test_word, model, args.model_type, args.device, max_wrong_guesses=6, verbose=2)
    
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_words = [line.strip() for line in f]
        eval_subset(model, args.model_type, args.device, test_words)

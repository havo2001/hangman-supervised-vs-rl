## Problem
Train a Transformer that can guess the next letter in any partially-revealed Hangman word, given that all target words come from one fixed dictionary.
## Approach:
### Supervised: Training a CNN-Transformers
1. For each word in the dictionary, I randomly generate missing characters with probabilities 0.2, 0.3, 0.4, 0.5, and 0.6, deciding for every character whether it should be hidden or not. This is essential because, at the beginning of the game, we have little information about the other letters and it also helps enrich the dataset. After that, I run through each character and treat it as a guessing character. I set `MAX_LEN = 40` for the longest word and convert every sample into a vector of length 40; words that are shorter are padded to this length.
2. Then I train a CNN-transformer on this dataset.

```
python -m utils.cnn_transformers_create_training_examples \
  --file data/train_data.txt \
  --output_file dataset 
```


```
python -m training.train_cnn_transformers \
--dataset data/dataset.npz \
--epochs 6 \
--batch_size 512
```

### Reinforcement Learning: TDQN Approach for Hangman
I formulate Hangman as a reinforcement learning problem and train a Deep Q-Network (DQN) to learn a letter-guessing policy (inspired by the PyTorch DQN tutorial and the original DQN paper).

Each episode samples a random word. The state includes the partially revealed word, guessed letters, and remaining guesses. The action space consists of 26 letters.

Rewards:
- +1 for correct guess  
- −1 for incorrect guess  
- +10 for completing the word  
- −10 for failure  

The model is trained with experience replay and a target network to learn the action-value function \( Q(s, a) \).

```
python -m training.train_dqn \
--replay_capacity 100 \
--device "cuda" \
--batch_size 512 \
--num_episodes 10 
```

## Results
If you're interested in the results and inference process, see:
`notebooks/hangman_solver_conv_transformer_results.ipynb`

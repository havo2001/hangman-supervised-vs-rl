## Problem
Train a Transformer that can guess the next letter in any partially-revealed Hangman word, given that all target words come from one fixed dictionary.
## Approach:
1. For each word in the dictionary, I randomly generate missing characters with probabilities 0.2, 0.3, 0.4, 0.5, and 0.6, deciding for every character whether it should be hidden or not. This is essential because, at the beginning of the game, we have little information about the other letters and it also helps enrich the dataset. After that, I run through each character and treat it as a guessing character. I set `MAX_LEN = 40` for the longest word and convert every sample into a vector of length 40; words that are shorter are padded to this length.
2. Then I train a CNN-transformer on this dataset.

```
python create_training_examples.py \
--file data/train_data.txt \
--output_file dataset 
```

```
python train.py          
--dataset dataset.npz \   
--epochs 6 \            
--batch_size 512 
```

## Result
The result is demonstrated in ```hangman_solver_conv_transformer_results.ipynb```.

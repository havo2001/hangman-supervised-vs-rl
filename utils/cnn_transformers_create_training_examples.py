import numpy as np
import random
import argparse

# Here / for padding, * for missing character, _ for masked character
ALPHABET = 'abcdefghijklmnopqrstuvwxyz*_/'
CHAR2ID  = {c:i for i,c in enumerate(ALPHABET)}
ID2CHAR  = {i:c for i,c in enumerate(ALPHABET)}
UNK_ID, MASK_ID, PAD_ID = 26, 27, 28
MASK_CHAR = '_'
MAX_LEN = 40


def simulated_missing_char(word, probs=(0.6, 0.5, 0.4, 0.3, 0.2)):
    """Return ≤ len(probs) variants of `word` where letters are replaced by '*'.
    Duplicates are automatically removed by the set."""
    variants = set()
    for p in probs:
        sim = ''.join('*' if random.random() < p else ch for ch in word)
        variants.add(sim)
    return variants 


def create_single_masked_word(word: str, idx: int):
    """
    Return (token_ids, label) for a single masked example or None if
    the word is out of length bounds.

    word : original string
    idx  : 0-based position to mask  (must be in range)
    """
    if not (0 <= idx < len(word)):
        raise IndexError("mask_idx out of range")

    if not (3 <= len(word) <= MAX_LEN):
        return None

    masked = list(word)
    masked[idx] = MASK_CHAR                       # "_"
    padded = ''.join(masked).ljust(MAX_LEN, '/')
    token = [CHAR2ID.get(c, UNK_ID) for c in padded]
    label  = CHAR2ID[word[idx]]   
    pad_mask = [pos == PAD_ID for pos in token]  # True for padding positions

    return token, label, idx, pad_mask


def simulate_dataset(words):
    tok_rows, labels, mask_pos, pad_mask_rows = [], [], [], []

    for word in words:
        if not (2 <= len(word) <= MAX_LEN):
            continue

        for i, ch in enumerate(word):
            if not ch.isalpha():
                continue

            example = create_single_masked_word(word, i)
            if example is None:
                continue
            token, label, idx, pad_mask = example
            tok_rows.append(token)
            labels.append(label)
            mask_pos.append(idx)
            pad_mask_rows.append(pad_mask)

    return (np.asarray(tok_rows, dtype=np.uint8),
            np.asarray(labels, dtype=np.uint8),
            np.asarray(mask_pos, dtype=np.uint8),
            np.asarray(pad_mask_rows, dtype=np.bool_))


def create_dataset(file, output_file:str):
    with open(file) as f:
        word_list = f.read().splitlines()

    # Simulate missing characters
    simulated_words = []
    for word in word_list:
        simulated_words.extend(simulated_missing_char(word))

    # Create dataset
    toks, labels, mas_pos, pad_mask = simulate_dataset(simulated_words)

    # Save dataset
    np.savez(f"data/{output_file}.npz", toks=toks, labels=labels, mas_pos=mas_pos, pad_mask=pad_mask)
    print(f'Dataset saved with {len(toks)} samples.')
    n = 100
    print(f'Sample {n}:', toks[n].T, 
          'Label:', labels[n], 
          'Masked position:', mas_pos[n],
          'Masked character:', ID2CHAR[labels[n]],
          'Pad mask:', pad_mask[n].T)
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='File that we will use as word dictionary to create the dataset')
    ap.add_argument('--output_file', default='dataset', help='Output file name')
    args = ap.parse_args()
    create_dataset(args.file, args.output_file)

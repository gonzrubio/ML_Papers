# -*- coding: utf-8 -*-
"""
CTC Key Word Spotter Using CTC
"""

import numpy as np
from collections import deque


def ctc_forward_pass(logits, target_labels, blank=0):
    """
    Calculates the alpha matrix for a single sequence.
    This is the forward pass of CTC.
    
    Args:
        logits (np.ndarray): The network outputs. Shape: (T, C)
        target_labels (np.ndarray): The ground truth labels. Shape: (S)
        blank (int): The index of the blank label.

    Returns:
        np.ndarray: The alphas array.
    """
    T, C = logits.shape
    S = len(target_labels)
    
    l_prime = np.full(2 * S + 1, blank)
    l_prime[1::2] = target_labels
    S_prime = len(l_prime)

    alphas = np.full((T, S_prime), -np.inf)
    alphas[0, 0] = logits[0, blank]
    if S_prime > 1:
        alphas[0, 1] = logits[0, l_prime[1]]

    def log_sum_exp(*args):
        if all(a == -np.inf for a in args):
            return -np.inf
        a_max = np.max(args)
        return a_max + np.log(np.sum([np.exp(a - a_max) for a in args]))

    for t in range(1, T):
        for s in range(S_prime):
            current_char = l_prime[s]
            term1 = alphas[t-1, s]
            term2 = alphas[t-1, s-1] if s > 0 else -np.inf
            term3 = alphas[t-1, s-2] if s > 1 and l_prime[s] != l_prime[s-2] else -np.inf
            alphas[t, s] = log_sum_exp(term1, term2, term3) + logits[t, current_char]

    return alphas


class KeyWordSpotter:
    """
    An online key word spotter using a sliding buffer and CTC.
    """
    def __init__(self, key_word, buffer_size, char_map):
        """
        Args:
            key_word (str): The word to spot.
            buffer_size (int): The maximum length of the input buffer.
            char_map (dict): Mapping from characters to their integer indices.
        """
        self.buffer_size = buffer_size
        self.char_map = char_map
        self.key_word_labels = np.array([self.char_map[c] for c in key_word])
        self.buffer = deque(maxlen=self.buffer_size)

    def process_frame(self, log_probs_frame):
        """
        Processes a single frame of RNN output and updates the spotting probability.

        Args:
            log_probs_frame (np.ndarray): Log probabilities for the current time step. Shape: (C,)

        Returns:
            float: The updated log probability of the key word.
        """
        # Add the new frame to the buffer
        self.buffer.append(log_probs_frame)

        # If the buffer isn't full yet, we can't get a meaningful result.
        if len(self.buffer) < len(self.key_word_labels):
            return -np.inf

        # Convert buffer to a numpy array for the CTC forward pass
        buffer_array = np.array(list(self.buffer))
        
        # Run the CTC forward pass on the buffer
        alphas = ctc_forward_pass(buffer_array, self.key_word_labels)

        # Calculate the log probability of the key word
        # This is the sum of the last two elements of the last row of alphas
        # representing paths ending in the last character or a blank.
        log_prob_final = self._log_sum_exp(alphas[-1, -1], alphas[-1, -2])

        return log_prob_final

    def _log_sum_exp(self, a, b):
        """
        Numerically stable log-sum-exp helper.
        """
        if a == -np.inf and b == -np.inf:
            return -np.inf
        a_max = max(a, b)
        return a_max + np.log(np.exp(a - a_max) + np.exp(b - a_max))


# Example Usage
if __name__ == '__main__':
    # a simple character map for our vocabulary
    char_map = {
        '-': 0, 'h': 1, 'e': 2, 'l': 3, 'o': 4,
        'a': 5, 'b': 6, 'c': 7, 'd': 8, 'w': 9
    }
    
    # Define the key word to spot
    key_word = "hello"
    buffer_size = 10 # sliding window will be 10 time steps long

    spotter = KeyWordSpotter(key_word, buffer_size, char_map)

    # --- Simulate RNN output over time ---
    
    # The RNN output is a probability distribution over all characters
    # We'll represent this with a numpy array for each time step.
    
    # 1. Start with some noise
    print("Simulating noise...")
    for t in range(5):
        # Create a random log-prob frame
        frame = np.random.randn(len(char_map))
        log_prob = spotter.process_frame(frame)
        print(f"Time: {t+1}, Key Word Probability: {np.exp(log_prob):.4f}")

    # 2. Add the word "hello" in the middle of the stream
    print("\nSimulating the word 'hello'...")
    word_frames = [
        [0.1, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 'h'
        [0.1, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 'e'
        [0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 'l'
        [0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 'l'
        [0.1, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]  # 'o'
    ]
    for t, probs in enumerate(word_frames):
        # Convert simple probs to log-probs
        log_probs = np.log(np.array(probs) + 1e-9)
        log_prob = spotter.process_frame(log_probs)
        print(f"Time: {t+6}, Key Word Probability: {np.exp(log_prob):.4f}")

    # 3. Add more noise to the end
    print("\nSimulating more noise...")
    for t in range(5):
        frame = np.random.randn(len(char_map))
        log_prob = spotter.process_frame(frame)
        print(f"Time: {t+11}, Key Word Probability: {np.exp(log_prob):.4f}")

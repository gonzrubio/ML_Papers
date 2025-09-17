# -*- coding: utf-8 -*-
"""
Connectionist Temporal Classification:
    Labelling Unsegmented Sequence Data with Recurrent Neural Networks

Implementation as descibed in: https://www.cs.toronto.edu/~graves/icml_2006.pdf

This implementation supports both forward and backward propagation operations.
"""

import numpy as np


def log_sum_exp(*args):
    """
    Computes log(sum(exp(x))) in a numerically stable way.
    """
    if not args:
        return -np.inf
    args = [x for x in args if x != -np.inf]
    if not args:
        return -np.inf
    
    max_val = max(args)
    sum_of_exps = sum(np.exp(x - max_val) for x in args)
    return max_val + np.log(sum_of_exps)


def _ctc_calculate_gradient(alphas, betas, lp, l_prime, log_prob_final, input_len, S_prime):
    """
    Calculates the gradient of the CTC loss using the forward and backward variables.
    
    Returns:
        np.ndarray: The gradient for a single sequence.
    """
    probs = np.exp(lp)
    grad_i = np.zeros_like(lp)
    
    for t in range(input_len):
        for s in range(S_prime):
            char_idx = l_prime[s]
            log_alpha_beta = alphas[t, s] + betas[t, s]
            if log_alpha_beta != -np.inf:
                grad_i[t, char_idx] -= np.exp(log_alpha_beta - log_prob_final)
                
    return probs + grad_i


def _ctc_backward_pass(lp, l_prime, input_len, S_prime, blank):
    """
    Performs the backward pass of the CTC algorithm to compute beta variables.
    
    Returns:
        np.ndarray: The beta variables.
    """
    betas = np.full((input_len, S_prime), -np.inf)
    betas[-1, -1] = lp[-1, blank]
    if S_prime > 1:
        betas[-1, -2] = lp[-1, l_prime[-2]]
    
    for t in range(input_len - 2, -1, -1):
        for s in range(S_prime):
            if s > 2 * t:
                continue
            
            term1 = betas[t+1, s]
            term2 = -np.inf
            if s < S_prime - 1:
                term2 = betas[t+1, s+1]
            term3 = -np.inf
            if s < S_prime - 2 and l_prime[s] != l_prime[s+2]:
                term3 = betas[t+1, s+2]
            
            betas[t, s] = log_sum_exp(term1, term2, term3) + lp[t, l_prime[s]]
    
    return betas


def _ctc_forward_pass(lp, l_prime, input_len, S_prime, blank):
    """
    Performs the forward pass of the CTC algorithm to compute alpha variables.
    
    Returns:
        tuple: (alphas, log_prob_final)
    """
    alphas = np.full((input_len, S_prime), -np.inf)
    alphas[0, 0] = lp[0, blank]
    if S_prime > 1:
        alphas[0, 1] = lp[0, l_prime[1]]
    
    for t in range(1, input_len):
        for s in range(S_prime):
            if s < S_prime - 2 * (input_len - t) - 1:
                continue
            
            term1 = alphas[t-1, s]
            term2 = -np.inf
            if s > 0:
                term2 = alphas[t-1, s-1]
            term3 = -np.inf
            if s > 1 and l_prime[s] != l_prime[s-2]:
                term3 = alphas[t-1, s-2]
            
            alphas[t, s] = log_sum_exp(term1, term2, term3) + lp[t, l_prime[s]]
    
    log_prob_final = log_sum_exp(alphas[-1, -1], alphas[-1, -2])
    return alphas, log_prob_final


def ctc_forward_backward(logits, targets, input_lengths, target_lengths, blank=0):
    """
    Computes the CTC loss and gradient for a batch of sequences.
    
    This function implements the forward and backward passes.
    
    Args:
        logits (np.ndarray): The unnormalized network outputs. Shape: (T, N, C)
        targets (np.ndarray): The target label sequence. Shape: (N, S)
        input_lengths (np.ndarray): Lengths of the input sequences. Shape: (N,)
        target_lengths (np.ndarray): Lengths of the target sequences. Shape: (N,)
        blank (int): The index of the blank label.
    
    Returns:
        tuple: (loss, gradient)
            loss (float): The total CTC loss.
            gradient (np.ndarray): The gradient of the loss with respect to the logits.
                                  Shape: (T, N, C)
    """
    N, S = targets.shape
    T, _, C = logits.shape
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=2, keepdims=True))
    grads = np.zeros_like(logits)
    total_loss = 0

    for i in range(N):
        input_len = input_lengths[i]
        target_len = target_lengths[i]
        target = targets[i, :target_len]
        lp = log_probs[:input_len, i, :]
        
        # Prepare modified target sequence (l')
        l_prime = np.full(2 * target_len + 1, blank)
        l_prime[1::2] = target
        S_prime = len(l_prime)

        # Forward Pass
        alphas, log_prob_final = _ctc_forward_pass(lp, l_prime, input_len, S_prime, blank)
        
        # Backward Pass
        betas = _ctc_backward_pass(lp, l_prime, input_len, S_prime, blank)
        
        # Calculate Gradient
        grad_i = _ctc_calculate_gradient(alphas, betas, lp, l_prime, log_prob_final, input_len, S_prime)
        
        total_loss -= log_prob_final
        grads[:input_len, i, :] = grad_i
    
    return total_loss, grads


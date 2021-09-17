#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:31:50 2021

@author: gonzo
"""

# MLP class
# - two hidden layers with ReLU activations
# - followed by a non-activated uputput layer with size 129
# - LayerNorm (all except decoder)


# Encoder class
# - knn with "connectivity radious" R to contruct the graph
# - graph edges are recomputed at every time step to reflect current positions
# - graph and edge encoders are MLPs
# - dimension of encodings is 128
# - Relative node encoder ignores position info by masking it out.
# - Relative edge encoder is provided with the relative positional displacement
# and its magnitude rij = [delta pij, mag delta pij]
# Concatenate global properties onto each particle state vector before passing
# it to the node relative encoder.


# Processor class
# - Stack of M=10 GCNs
# - MLPs as internal edge and node update functions (shared or unshared params)
# - GCNs without global features or updates, and with residual connections 
# between the input and output latent node and edge attributes


# Decoder class
# - Decoder is a MLP
# - After decoder, the future position and velocity are updated using an Euler
# integrator so the (supervised training) output corresponds to accelerations


# simulator class (encode-process-decode)


# Data
# input state vector [position, vel_1, ..., vel_5, material_features]
# prediction targets are the per-particle average acceleration.
# velocity and acceleration are computed from the position using finite
# differences.
# At training add random-walk noise to the training set 3e-4
# Standardize all input and target vectors


# Training
# - L2-norm of learned dynamics and ground truth accelerations (remove noise)
# - Addam optimizer
# - Exponential learning rate decay from e-4 to e-6








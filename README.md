# Vanilla GCN on Cora: A From-Scratch Implementation

This repository contains a semi-supervised node classification model built on the Cora citation network. Instead of relying on black-box library layers like PyTorch Geometric's `GCNConv`, this project implements the Kipf & Welling Vanilla GCN propagation rule entirely from scratch in pure PyTorch.

By explicitly constructing the symmetric normalized adjacency matrix ($\hat{A}$) and defining the layer operations as raw matrix multiplications ($\hat{A}HW$), this implementation exposes the actual mathematical engine of the network.

## Key Highlights
* **No Pre-Built Layers:** The graph convolution is written from the ground up to demonstrate the mechanics of neighborhood aggregation and feature transformation.
* **Baseline Accuracy:** Achieves **81.2% test accuracy** using only 140 labeled nodes, matching the original published benchmarks.
* **Lightweight Architecture:** A 2-layer `bias=False` design resulting in exactly 23,040 trainable parameters.
* **Over-Smoothing Analysis:** Includes a dedicated experiment and visualization proving how stacking deep GCN layers acts as a low-pass filter, collapsing node embeddings via floating-point underflow.
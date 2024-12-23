import torch
from torch import nn
import numpy as np

def apply_kmeans_quantization(weights, n_clusters=256):
    shape = weights.shape
    flattened_weights = weights.flatten().float()
    with torch.no_grad():
        # Using k-means to find centroids and labels
        centroids, labels = torch.kmeans(flattened_weights, num_clusters=n_clusters, iter=10)
        quantized_weights = centroids[labels].view(shape)
    return quantized_weights

def apply_linear_quantization(weights, levels=256):
    min_val, max_val = weights.min(), weights.max()
    scale = (max_val - min_val) / (levels - 1)
    zero_point = min_val
    quantized_weights = torch.round((weights - zero_point) / scale) * scale + zero_point
    return quantized_weights

def quantization_aware_training(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
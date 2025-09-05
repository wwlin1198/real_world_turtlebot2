import torch
import numpy as np
import pickle
import os
import argparse

def extract_weights(model_path, output_path):
    # Load PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract weights into a dictionary
    weights = {}
    
    # Extract CNN weights
    cnn_layers = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
    weights['cnn_w1'] = cnn_layers[0].weight.data.numpy()
    weights['cnn_b1'] = cnn_layers[0].bias.data.numpy()
    weights['cnn_w2'] = cnn_layers[1].weight.data.numpy()
    weights['cnn_b2'] = cnn_layers[1].bias.data.numpy()
    weights['cnn_w3'] = cnn_layers[2].weight.data.numpy()
    weights['cnn_b3'] = cnn_layers[2].bias.data.numpy()
    
    # Extract FC layer weights
    fc_layers = [module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    weights['fc_w1'] = fc_layers[0].weight.data.numpy()
    weights['fc_b1'] = fc_layers[0].bias.data.numpy()
    weights['fc_w2'] = fc_layers[1].weight.data.numpy()
    weights['fc_b2'] = fc_layers[1].bias.data.numpy()
    weights['output_w'] = fc_layers[2].weight.data.numpy()
    weights['output_b'] = fc_layers[2].bias.data.numpy()
    
    # Save weights to file
    with open(output_path, 'wb') as f:
        pickle.dump(weights, f, protocol=2)  # Use protocol 2 for Python 2.7 compatibility
    
    print(f"Saved weights to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save weights")
    args = parser.parse_args()
    
    extract_weights(args.model_path, args.output_path)
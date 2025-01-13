from datasets import load_dataset
import numpy as np
import torch

# Load the Fashion-MNIST dataset
dataset = load_dataset('fashion_mnist')

# Convert the images to numpy arrays
train_images = np.array(dataset['train']['image'])
test_images = np.array(dataset['test']['image'])

# Convert the numpy arrays to tensors
train_images_tensor = torch.tensor(train_images, dtype=torch.float32)
test_images_tensor = torch.tensor(test_images, dtype=torch.float32)

# Flatten the tensors
train_images_flattened = train_images_tensor.view(train_images_tensor.size(0), -1)
test_images_flattened = test_images_tensor.view(test_images_tensor.size(0), -1)

print("Flattened train images shape:", train_images_flattened.shape)
print("Flattened test images shape:", test_images_flattened.shape)
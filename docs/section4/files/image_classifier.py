#!/usr/bin/env python3
##
# SOURCE https://pytorch.org/vision/0.19/models.html
##

from torchvision.io import read_image
from torchvision.models import resnet101, ResNet101_Weights
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("image", help="the image to classify (str)", type=str)
args = parser.parse_args()

image_path = args.image
if not os.path.exists(image_path):
    print(f"Error: file not found: {image_path}")
    sys.exit(1)

# 1 - Initialize model with best available weights
weights = ResNet101_Weights.DEFAULT
model = resnet101(weights=weights)
model.eval()

# 2 - Initialize the inference transforms
preprocess = weights.transforms()

print(f"Classifying {image_path} with ResNet101...")
img = read_image(f"{image_path}")

# 3 - Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# 4 - Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

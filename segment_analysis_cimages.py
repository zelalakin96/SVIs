!pip install torch torchvision opencv-python numpy pandas tqdm

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Required imports
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# PSPNet model class
class PSPNet(torch.nn.Module):
    def __init__(self, n_classes=150):
        super(PSPNet, self).__init__()
        # Using ResNet18 as backbone for memory efficiency
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.classifier = torch.nn.Conv2d(512, n_classes, kernel_size=1)
        
    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        return self.classifier(features)

def load_model(model_path=None):
    """Load PSPNet model with memory optimization"""
    model = PSPNet(n_classes=150)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def process_image(model, image, device, mean, std):
    """Process a single image with memory optimization"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize image
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    # Convert to tensor and move to device
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    image = image.unsqueeze(0).to(device)
    
    # Process image
    with torch.no_grad():
        output = model(image)
        output = F.interpolate(output, size=(640, 640), mode='bilinear', align_corners=False)
        prediction = output.argmax(1)
    
    return prediction.cpu().numpy()[0]

def main():
    # Set up device (GPU if available)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Set up Google Drive directories
    input_dir = Path('/content/drive/MyDrive/your_input_folder')  # Change this to your input folder
    output_dir = Path('/content/drive/MyDrive/your_output_folder')  # Change this to your output folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Load model
    logging.info("Loading model...")
    model = load_model()
    model = model.to(DEVICE)
    
    # Process images
    image_files = list(input_dir.glob('*.jpg'))  # or '*.png' etc.
    total_images = len(image_files)
    logging.info(f"Found {total_images} images to process")
    
    # Process each image
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                logging.error(f"Could not load image: {img_path}")
                continue
                
            # Process image
            prediction = process_image(model, image, DEVICE, mean, std)
            
            # Save segmentation mask
            output_path = output_dir / f"{img_path.stem}_seg.png"
            cv2.imwrite(str(output_path), prediction)
            
            # Calculate class distribution
            class_distribution = np.zeros(150)
            for i in range(150):
                class_distribution[i] = np.mean(prediction == i)
            
            # Store results
            results.append({
                'image_name': img_path.name,
                'class_distribution': class_distribution
            })
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"Error processing {img_path}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv('/content/drive/MyDrive/segmentation_results.csv', index=False)
        logging.info("Results saved to segmentation_results.csv")
    
    logging.info("Processing completed!")

if __name__ == '__main__':
    main()

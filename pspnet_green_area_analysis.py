import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PSPNet için open-mmlab/mmsegmentation veya başka bir repo gereklidir.
# Burada örnek olması için DeepLabV3+ kullanıldı. PSPNet ile değiştirmek için ilgili modeli yükleyin.
from torchvision.models.segmentation import deeplabv3_resnet101

def load_model():
    model = deeplabv3_resnet101(pretrained=True).eval()
    return model

def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0)

def analyze_green_area(mask, green_classes):
    mask_np = mask.squeeze().cpu().numpy()
    green_mask = np.isin(mask_np, green_classes)
    green_ratio = green_mask.sum() / mask_np.size
    return green_ratio

def main(image_folder):
    model = load_model()
    green_classes = [21, 22]  # COCO: 21=grass, 22=tree (DeepLabV3+ için)
    results = {}
    # Çocuk verilerini oku
    df = pd.read_excel('217_cocuk_veri.xlsx')
    # StudentID ile eşleşen görsellerin yeşil alan ortalamasını ekle
    green_areas = []
    for idx, row in df.iterrows():
        student_id = row['ID']
        ratios = []
        for heading in [0, 90, 180, 270]:
            fname = f"street_view_{student_id}_lat"
            # İlgili görseli bul
            matches = [f for f in os.listdir(image_folder) if f.startswith(fname) and f"heading{heading}" in f]
            for img_file in matches:
                img_path = os.path.join(image_folder, img_file)
                input_tensor = preprocess_image(img_path)
                with torch.no_grad():
                    output = model(input_tensor)['out']
                mask = output.argmax(1)
                green_ratio = analyze_green_area(mask, green_classes)
                ratios.append(green_ratio)
        # Ortalama yeşil alan oranı
        if ratios:
            green_areas.append(np.mean(ratios))
        else:
            green_areas.append(np.nan)
    df['YESIL_ALAN_ORANI'] = green_areas
    df.to_excel('217_cocuk_veri_yesil_alan.xlsx', index=False)
    print('Excel dosyasına YESIL_ALAN_ORANI sütunu eklendi ve kaydedildi.')

if __name__ == "__main__":
    main('street_view_images')

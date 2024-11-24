import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class ChildSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")])
        self.mask_paths = sorted([os.path.join(mask_dir, mask) for mask in os.listdir(mask_dir) if mask.endswith(".jpg")])
        self.transform = transform

        # Debugging prints
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        print(f"Number of images: {len(self.image_paths)}")
        print(f"Number of masks: {len(self.mask_paths)}")

        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:
            raise ValueError("No images or masks found in the specified directories.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Convert to grayscale
        mask = np.array(mask)  # Convert to numpy array
        mask = torch.tensor(mask, dtype=torch.long)  # Convert to tensor and ensure it is of type long

        if self.transform:
            image = self.transform(image)

        return image, mask

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

image_dir = "jaun_dataset_Copy/Normal"
mask_dir = "jaun_masks_Copy/Normal"

train_dataset = ChildSegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
num_classes = 21  # Adjust this based on the number of classes in your masks
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'segmentation_model.pth')

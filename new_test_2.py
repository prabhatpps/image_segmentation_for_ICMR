import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torch.utils.data import DataLoader

# Define the hook function for debugging
def hook_fn(module, input, output):
    print(f"{module.__class__.__name__} output size: {output.size()}")

# Custom DeepLabV3 model
class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDeepLabV3, self).__init__()
        self.model = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # Register hooks for debugging
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                layer.register_forward_hook(hook_fn)

    def forward(self, x):
        x = self.model.backbone(x)['out']
        x = self.model.classifier(x)
        return x

# Initialize model
num_classes = 21  # Adjust based on your mask labels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomDeepLabV3(num_classes=num_classes)
model = model.to(device)

# Dummy DataLoader (Replace with your actual DataLoader)
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = torch.randn(3, 256, 256)  # Replace with actual data loading
        masks = torch.randint(0, num_classes, (256, 256))  # Replace with actual mask loading
        return inputs, masks

dataloader = DataLoader(DummyDataset(10), batch_size=4, shuffle=True)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, masks in dataloader:
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Debugging print statements
        print(f"Input size: {inputs.size()}")
        print(f"Output size: {outputs.size()}")
        print(f"Mask size: {masks.size()}")

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training complete")

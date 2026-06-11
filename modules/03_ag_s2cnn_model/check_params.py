from models.ag_s2cnn import AG_S2CNN
import torch

model = AG_S2CNN()
print('Model parameter breakdown:')
total = 0
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    total += params
    print(f'{name}: {params:,} ({params/1e6:.2f}M)')
print(f'Total: {total:,} ({total/1e6:.2f}M)')

# Check classifier specifically
print('\nClassifier breakdown:')
for name, param in model.classifier.named_parameters():
    print(f'{name}: {param.shape} = {param.numel():,}')

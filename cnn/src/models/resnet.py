import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18


def init_resnet(device, lr):
    model = resnet18(weights='IMAGENET1K_V1')

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        4,                      # 输入通道改成 4
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # 初始化新 conv1 的权重
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = old_conv.weight
        model.conv1.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, 1)

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

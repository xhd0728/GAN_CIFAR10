import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models import Generator, Discriminator

batch_size = 64
lr = 0.0002
betas = (0.5, 0.999)
epochs = 20

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator(in_channels=100, out_channels=3).to(device)
D = Discriminator(in_channels=3).to(device)
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
criterion = nn.BCELoss()

for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # print("type: ", type(real_images))
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        z = torch.randn(batch_size, 100, 32, 32).to(device)
        fake_images = G(z)
        outputs = D(fake_images.detach())

        # print("real: ",real_images.shape)
        # print("fake: ",fake_images.shape)

        d_loss_fake = criterion(outputs, fake_labels)
        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if (i+1) % 1 == 0:
            print(
                f'epoch: {epoch+1}/{epochs}, '
                f'step: {i+1}/{len(dataloader)}, '
                f'd_loss: {d_loss.item():.4f}, '
                f'g_loss: {g_loss.item():.4f}, '
                f'd_lr: {lr}, '
                f'g_lr: {lr}'
            )
    if (epoch+1) % 10 == 0:
            torch.save(G.state_dict(), f'./model/generator_{epoch+1}.pth')
            torch.save(D.state_dict(), f'./model/discriminator_{epoch+1}.pth')

print('ok')

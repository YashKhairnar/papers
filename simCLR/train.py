import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from sklearn.manifold import TSNE
from simCLR.models import ResNetSimCLR
from simCLR.datasets import SimCLRViewGenerator, get_simclr_transform, get_linear_eval_transform
from simCLR.losses import NTXentLoss


def visualize_representations(model, device, epoch):
    print(f"Visualizing representations for epoch {epoch}...")
    model.eval()
    # Use standard eval transform for single-view visualization
    transform = get_linear_eval_transform(train=False)
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    
    features = []
    labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h, _ = model(x)
            features.append(h.cpu().numpy())
            labels.append(y.numpy())
            if len(features) * 512 >= 2000: # 2000 points is enough for a good t-SNE plot
                break
                
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f't-SNE Visualization - Epoch {epoch}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    plot_path = f'simCLR/plots/epoch_{epoch}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")


def train():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    transform = SimCLRViewGenerator(get_simclr_transform())
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

    model = ResNetSimCLR(out_dim=128).to(device)
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs_list = []
    losses_list = []

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        epochs_list.append(epoch + 1)
        losses_list.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.4f}')

        if (epoch + 1) % 2 == 0:
            # Checkpointing
            checkpoint_path = f'simCLR/checkpoints/checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Visualization
            visualize_representations(model, device, epoch + 1)

    # Save final loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, losses_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('simCLR/plots/final_loss_curve.png')
    plt.close()

    torch.save(model.state_dict(), 'simclr_pretrained.pth')


if __name__ == '__main__':
    if not os.path.exists("simCLR/checkpoints"):
        os.makedirs("simCLR/checkpoints")
    if not os.path.exists("simCLR/plots"):
        os.makedirs("simCLR/plots")
    train()
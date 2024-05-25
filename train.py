from diffusion_utils import *
from noise_estimation_unet import NEU
from torch.utils.data import DataLoader
from data_loader import SARDataLoader


def main():
    train_data = SARDataLoader("./data/train", data ="palsar", train=True)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True)

    model = NEU()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001)

    for epoch in range(epochs):
        print('epoch={}'.format(epoch))
        model.train()
        for idx, (x, label) in enumerate(train_data_loader):
            x, label = x.to(device), label.to(device)
            t = torch.randint(0, all_time_steps, (batch_size,)).long()
            mse_loss = compute_losses(model, x, label, t)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            print('------epoch={}, idx={}, mse_loss={}'.format(epoch, idx, mse_loss))
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "./msd/palsar_{}.pth")

    print("Training completed!")


if __name__ == '__main__':
    main()

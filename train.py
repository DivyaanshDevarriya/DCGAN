import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator, Generator, initialize_weights
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_EPOCS = 10
SAVE_EPOCH = 3

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)])
    ]
)

# dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms, download = True)
dataset = datasets.ImageFolder(root = 'celeb_dataset', transform = transforms)
loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas = (0.5,0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas = (0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

step = 0

img_list = []
gen_losses = []
disc_losses = []

gen.train()
disc.train()

try: 
    for epoch in range(NUM_EPOCS):
        for batch_idx, (real, _) in enumerate(loader):
            real  = real.to(device)
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1 , 1)).to(device)
            fake = gen(noise)

            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)

            disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_loss_real + disc_loss_fake)/2
            
            disc.zero_grad()
            disc_loss.backward(retain_graph = True)
            opt_disc.step()


            output = disc(fake).reshape(-1)
            gen_loss = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCS} Batch {batch_idx}/{len(loader)}] Loss D : {disc_loss: .4f}, Loss G : {gen_loss: .4f}")
                
                with torch.no_grad():
                    fake = gen(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize = True
                    )

                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize = True
                    )

                    writer_real.add_image("Real", img_grid_real, global_step=step)
                    writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                step+=1

            gen_losses.append(round(gen_loss.item(), 4))
            disc_losses.append(round(disc_loss.item(), 4))
        
        if (epoch % 2 == 0) or (epoch == NUM_EPOCS-1):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
            img_grid = torchvision.utils.make_grid(
                            fake[:32], normalize = True
                        )
            img_list.append(img_grid)

        if epoch % SAVE_EPOCH == 0:
            torch.save({
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'optim_gen': opt_gen.state_dict(),
                'optim_disc': opt_disc.state_dict(),
                'params': {'CHANNELS_IMG': CHANNELS_IMG, 'FEATURES_GEN': FEATURES_GEN, 'FEATURES_DISC': FEATURES_DISC, 'Z_DIM': Z_DIM}
            }, f'model/model_epoch_{epoch}.pth')

    torch.save({
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(),
        'optim_gen': opt_gen.state_dict(),
        'optim_disc': opt_disc.state_dict(),
        'params': {'CHANNELS_IMG': CHANNELS_IMG, 'FEATURES_GEN': FEATURES_GEN, 'FEATURES_DISC': FEATURES_DISC, 'Z_DIM': Z_DIM}
    }, 'model/model_final.pth')    


    torch.cuda.empty_cache()

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_losses,label="G")
    plt.plot(disc_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('images/losses.png')
    plt.show()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()
    anim.save('images/celeba.gif', dpi=80, writer='pillow')
except Exception as e:
    print(e)
    torch.cuda.empty_cache()


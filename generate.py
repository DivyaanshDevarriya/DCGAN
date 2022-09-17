import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from dcgan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='model/model_final.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
args = parser.parse_args()

state_dict = torch.load(args.load_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

params = state_dict['params']
Z_DIM = params['Z_DIM']
CHANNELS_IMG = params['CHANNELS_IMG']
FEATURES_GEN = params['FEATURES_GEN']


gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
gen.load_state_dict(state_dict['generator'])

noise = torch.randn(int(args.num_output), params['Z_DIM'], 1, 1).to(device)

with torch.no_grad():
    generated_img = gen(noise).detach().cpu()

plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1,2,0)))
plt.savefig('images/generated_img.png')
plt.show()
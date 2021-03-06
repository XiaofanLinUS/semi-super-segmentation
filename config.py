import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f'we are using {device} for training!')
# Dataloader Configure
loader_config = {'batch_size': 1,
                 'shuffle': True,
                 'num_workers': 1,
                 'sampler': None}

# Model Saving Path
SAVING_PATH = './model_25random_lrelu_r'

# Training Epoch
epoches = 80

mri_data = './mri_data'

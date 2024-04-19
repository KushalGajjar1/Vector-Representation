import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 32
epochs = 50

dimension = 14

filters_to_prune = [[0, 0, 0, 0],
                    [2, 4, 8, 16],
                    [4, 8, 16, 32],
                    [8, 16, 32, 64],
                    [16, 32, 64, 128],
                    [30, 50, 100, 200]]

# filters_to_prune = [[8, 16, 32, 64],
#                     [16, 32, 64, 128],
#                     [30, 50, 100, 200]]
import torch

checkpoint_url = "https://msravcghub.blob.core.windows.net/simmim-release/swinv2/pretrain/swinv2_small_1kper10_125k.pth"

state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")['model']

for name, param in state_dict.items():
    print(name, param.shape)
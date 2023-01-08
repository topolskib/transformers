import pickle


filepath = "/Users/nielsrogge/Documents/Mask2Former_checkpoints"
with open(filepath, "rb") as f:
    data = pickle.load(f)

state_dict = data["model"]

for name, param in state_dict.items():
    print(name, param.shape)

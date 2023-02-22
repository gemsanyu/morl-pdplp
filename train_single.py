from torch.utils.data import DataLoader

from bpdplp.bpdplp_dataset import BPDPLP_Dataset

def run():
    dataset = BPDPLP_Dataset(num_requests=20)
    dl = DataLoader(dataset, batch_size=64, num_workers=4)
    for i, batch in enumerate(dl):
        print(i)

if __name__ == "__main__":
    run()
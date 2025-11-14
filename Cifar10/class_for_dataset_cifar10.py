
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle

class myDataset(Dataset):
  def __init__(self, root, train):
    self.images = []
    self.labels = []
    if train:
      data_files = [os.path.join(root, f"data_batch_{i}") for i in range(1,6)]
    else:
      data_files = [os.path.join(root, "test_batch")]
    # print(data_files)

    for data_file in data_files:
      with open(data_file, "rb") as f:
        batch = pickle.load(f, encoding = "bytes") # cần pickle.load vì bộ cifar là các file đã pickled
        self.images.extend(batch[b'data'])
        self.labels.extend(batch[b'labels'])
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.images[index].reshape(3,32,32).astype("float32")/255, self.labels[index]

if __name__ == "__main__":
  train_data = myDataset(root ="C:/Users/ASUS/Hoc_DL/learning-DL/Cifar10/Data/cifar-10-batches-py",train = True)
  data = DataLoader(
      dataset= train_data,
      batch_size = 10,
      shuffle = True,
      num_workers= 4,
      drop_last= True
  )

  for i,j in enumerate(data):
    print(i,j)
    break
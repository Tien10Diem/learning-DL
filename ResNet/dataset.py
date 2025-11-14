from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from torchvision.transforms import ToTensor
from PIL import Image


class myDataset(Dataset):
    def __init__(self, transform = None):
        self.list_labels = os.listdir(r'C:\Users\ASUS\Hoc_DL\learning-DL\Animal\animals')
        self.list_labels.sort()
        self.datas_file = []
        self.labels = []
        self.transform = transform
        for label in self.list_labels:
            path = os.path.join(r'C:\Users\ASUS\Hoc_DL\learning-DL\Animal\animals',label)
            
            for file in os.listdir(path):
                self.labels.append(label)
                self.datas_file.append(os.path.join(path, file))
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # ép tất cả ảnh về 3 kênh màu RGB
        image = Image.open(self.datas_file[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.list_labels.index(self.labels[index])
    
    def shapes(self, index):
        image, _ = self.__getitem__(index)
        return image.shape
    
             
if __name__ == "__main__":
    size = 224
    
    my_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        ToTensor(),
    ])
    
    data = myDataset(my_transforms)
    dataloader = DataLoader(
        dataset=data,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
   
    # # 1. Lấy "iterator" từ dataloader
    # data_iterator = iter(dataloader)
    # print(data_iterator)
    # # 2. Lấy batch đầu tiên từ iterator
    # # (Mỗi lần gọi 'next' sẽ lấy ra 1 batch)
    # images_batch, labels_batch = next(data_iterator)

    # # 3. Bây giờ bạn có thể in thông tin của batch đó
    # print("Kích thước của batch ảnh:", images_batch.shape)
    # print("Labels của batch:", labels_batch)
    # print("Số lượng ảnh trong batch:", len(labels_batch))
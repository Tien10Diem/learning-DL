from dataset import myDataset
from model import simpleCNN
import os
import torch
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import random_split
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import shutil
from torchvision.transforms import ToTensor

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--batch-size','-b', type=int, default=50, help='Batch size for training and testing')
    arg.add_argument('--epochs','-e', type=int, default=10, help='Number of training epochs')
    arg.add_argument('--size','-s', type=int, default=224, help='Image size (height and width)')
    arg.add_argument('--logging','-l', type = str, default = r'C:\Users\ASUS\Hoc_DL\learning-DL\ResNet\tensorboard', help = 'Tensorboard logging directory')
    arg.add_argument('--model', '-m', type = str, default = r'C:\Users\ASUS\Hoc_DL\learning-DL\ResNet\checkpoints', help = 'Model to train')
    arg.add_argument("--checkpoint", "-c", type=str, default=None)
    return arg.parse_args()

if __name__ == "__main__":
    args = get_args()
    writer = SummaryWriter(args.logging)
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    
    if os.path.exists(args.logging):
        shutil.rmtree(args.logging)
    os.makedirs(args.logging)
    
    best_acc = 0.0
    my_transforms = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        ToTensor(),
    ])
    data = myDataset(my_transforms)
    
    
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split
    import numpy as np

    data = myDataset(my_transforms)

    # Lấy tất cả indices và labels
    indices = np.arange(len(data))
    labels = [data.labels[i] for i in indices]

    # Stratified split - đảm bảo mỗi class đều có tỉ lệ train/test như nhau
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        stratify=labels,  # ✅ KEY: Giữ tỉ lệ class đều
        random_state=42
    )

    train_data = Subset(data, train_indices)
    test_data = Subset(data, test_indices)
    
    
    
    # train_size = int(0.8 * len(data))
    # test_size = len(data) - train_size
    # train_data, test_data = random_split(data, [train_size, test_size])
    
    
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    
    model = simpleCNN()
    
    if torch.cuda.is_available():
        print("Using GPU")
        model = model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_acc = 0
    
    from tqdm import tqdm
    for epoch in range(start_epoch, args.epochs):
        model.train()

        all_preds = []
        all_labels = []
        progress_bar = tqdm(train_dataloader)
        for iter, (img, label) in enumerate(progress_bar):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            
            output = model(img)
            loss = criterion(output, label)
  
            progress_bar.set_description(f"Epoch [{epoch+1}/{args.epochs}], Step [{iter+1}/{len(train_dataloader)}], Loss: {loss:.4f}")
            writer.add_scalar('Training/Loss', loss.item(), epoch * len(train_dataloader) + iter)
            
            # Xóa gradients cũ    
            optimizer.zero_grad()
            # Tính gradients mới
            loss.backward()
            # Cập nhật weights
            optimizer.step()

        model.eval()
        for iter, (img, label) in enumerate(test_dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            all_labels.extend(label.cpu().tolist())
            with torch.no_grad():
                preds = model(img)
                # Lấy nhãn dự đoán từ output của model
                indices = torch.argmax(preds, dim = 1)
                all_preds.extend(indices.cpu().tolist())
                
                loss = criterion(preds, label)
        acc = accuracy_score(all_labels, all_preds)
        print(classification_report(all_labels, all_preds))
        writer.add_scalar('Testing/Loss', loss.item(), epoch)
        torch.save(model.state_dict(),f"{args.model}/cnn.pt")
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc
            
        }
        torch.save(checkpoint, "{}/cnn.pt".format(args.model))
        if acc > best_acc:
            checkpoint = {
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/cnn_best.pt".format(args.model))
            best_acc = acc
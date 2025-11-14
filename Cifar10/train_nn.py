from class_for_dataset_cifar10 import myDataset
from modul import SimpleNN
import torch
from sklearn.metrics import classification_report

if __name__ == "__main__":
    train_data = myDataset("C:/Users/ASUS/Hoc_DL/learning-DL/Cifar10/Data/cifar-10-batches-py", True)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2, drop_last=True) 
    
    test_data = myDataset("C:/Users/ASUS/Hoc_DL/learning-DL/Cifar10/Data/cifar-10-batches-py", False)
    test_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2, drop_last=True) 
    
    model = SimpleNN(num_classes=10)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(10):
        model.train()

        all_preds = []
        all_labels = []
        for iter, (img, label) in enumerate(train_data_loader):
            # Forward pass
            output = model(img)
            loss = criterion(output, label)
            # print(f"Epoch [{epoch+1}/10], Step [{iter+1}/{len(train_data_loader)}], Loss: {loss:.4f}")
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        for iter, (img, label) in enumerate(test_data_loader):
            all_labels.extend(label.tolist())
            with torch.no_grad():
                preds = model(img)
                indices  = torch.argmax(preds, dim = 1)
                all_preds.extend(indices.tolist())
                loss = criterion(preds, label)

        print(classification_report(all_labels, all_preds))
import os
import time
import pickle
import numpy as np

from data import *
from utils import *
from model import TextClassifier

from torch import nn
from torch.utils.data import DataLoader

file_path = "/home/user19/bag/fastText_supervised/sentence_classification"
data_list = os.listdir(file_path)

# device = "cpu"
device = "cuda:0"
bigram = True
max_length = 256

final_result = dict()
st = time.time()

for data_type in data_list:
    print("-"*52,data_type,"-"*52)
    data_path = file_path + "/" + data_type
    sogou, lr = essential_args(data_type)
        
    load_dataset = GetTrainTestData(data_path, max_length, sogou, bigram)
    train_x = load_dataset.train_x
    train_y = load_dataset.train_y
    test_x = load_dataset.test_x
    test_y = load_dataset.test_y
        
    unique_labels = np.unique(np.array(train_y))
    label_size = len(unique_labels)
    label_to_id = dict()
    id_to_label = dict()
    la_id = 0
    for la in unique_labels:
        label_to_id[la] = la_id
        id_to_label[la_id] = la
        la_id += 1
    
    if bigram:
        train_dataset = ClassificationDataset(train_x, train_y, max_length, label_to_id, load_dataset.train_bigram_x)
        test_dataset = ClassificationDataset(test_x, test_y, max_length, label_to_id, load_dataset.test_bigram_x)
    else:
        train_dataset = ClassificationDataset(train_x, train_y, max_length, label_to_id)
        test_dataset = ClassificationDataset(test_x, test_y, max_length, label_to_id)
    
    max_epoch = 5
    train_data_size = len(train_y)
    batch_size = 64
    trainloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    
    vocab_size = len(load_dataset.word2id)
    hidden_size = 10
    label_size = len(np.unique(np.array(train_y)))
    
    model = TextClassifier(vocab_size, hidden_size, label_size).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr*batch_size)
    
    max_iter_per_epoch = train_data_size//batch_size + 1
    total_iters = max_epoch*max_iter_per_epoch + 1
    lambda_func = lambda iter: 1 - (iter / total_iters)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)
    
    train_loss = 0
    train_acc = 0
    print("#"*52,"TRAIN START","#"*52)
    for epoch in range(max_epoch):
        for iter, cache in enumerate(trainloader):
            x, y = cache
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            predict = model.forward(x)
            loss = loss_function.forward(predict, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_loss += loss.detach().cpu().item()
            
            _, predict = torch.max(predict, dim=1)
            correct = torch.eq(predict, y).sum().item()
            train_acc += correct/batch_size
            
            if iter%500 == 0:
                train_loss /= 500
                train_acc /= 500
                print(f"Epoch:{epoch:<5} iter:{iter:<6}/ {max_iter_per_epoch:<6} lr:{optimizer.param_groups[0]['lr']:<10.4f} Loss:{train_loss:<10.5f} Train acc:{train_acc*100:<10.3f} Time:{(time.time()-st)/3600:>6.4f} Hour")
                train_loss = 0
                train_acc = 0
        
        model.eval()
        with torch.no_grad():
            test_acc = 0
            for x, y in testloader:
                x = x.to(device)
                y = y.to(device)
                predict = model.forward(x)
                _, predict = torch.max(predict, dim=1)
                correct = torch.eq(predict, y).sum().item()
                test_acc += correct
            test_acc /= len(test_y)
            print(f"=======================================Data: {data_type} Epoch: {epoch} Test acc: {test_acc*100:.3f}=======================================")
        model.train()
    
    torch.save({
            'iter':iter,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"./save_model/{data_type}.pth")
    final_result[data_type] = test_acc

print(final_result)
with open("/home/user19/bag/fastText_supervised/result.pkl", "wb") as file:
    pickle.dump(final_result, file)
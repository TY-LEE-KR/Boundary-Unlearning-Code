import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from tqdm import tqdm
from models import AllCNN
import torch.nn.functional as F
import matplotlib.pyplot as plt


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def train(model, data_loader, criterion, optimizer, loss_mode, device='cpu'):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        # print(batch_y.size())

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)  # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y)  # mse loss
        elif loss_mode == "cross":
            # print(batch_y)
            # print(output)
            loss = criterion(output, batch_y)  # torch.argmax(batch_y, dim=1))  # cross entropy loss
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss


def train_save_model(train_loader, test_loader, model_name, optim_name, learning_rate, num_epochs, device, path):
    start = time.time()

    num_classes = max(train_loader.dataset.targets) + 1  # if args.num_classes is None else args.num_classes
    if model_name == 'AllCNN':
        model = AllCNN(n_channels=3, num_classes=num_classes, filters_percentage=0.5).to(device)

    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0

    for epo in range(num_epochs):
        print('EPOCH:{}'.format(epo))
        train(model=model, data_loader=train_loader, criterion=criterion, optimizer=optimizer, loss_mode='cross',
              device=device)
        _, acc = eval(model=model, data_loader=test_loader, mode='', print_perform=False, device=device)
        print('test acc:{}'.format(acc))

        if acc >= best_acc:
            best_acc = acc
            # state = {'net':model.state_dict(), 'features': feature_exct.state_dict()}
            torch.save(model, '{}.pth'.format(path))

    end = time.time()
    print('training time:', end-start, 's')
    return model


def test(model, loader):
    model.eval()
    outputavg = [0.] * 10
    cnt = [0] * 10
    res = ''
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(loader, leave=False)):
            # target = target.item()

            data = data.cuda()
            target = target.cuda()

            output = model(data)
            output = F.softmax(output, dim=-1).data.cpu().numpy().tolist()
            for i in range(len(target)):
                pred = target[i].cpu().int()
                if round(output[i][pred]) == 1:
                    outputavg[pred] += 1
                cnt[pred] += 1

    for i in range(len(outputavg)):
        if cnt[i] == 0:
            outputavg[i] = 0.
        else:
            outputavg[i] /= cnt[i]
        res += 'class {} acc: {:.2%}\n'.format(i, outputavg[i])
    return res


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.title('{} confusion matrix'.format(name), loc='center')
        plt.show()

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc

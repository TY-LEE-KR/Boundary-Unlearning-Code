import copy
import utils
from trainer import eval, loss_picker, optimizer_picker
import numpy as np
import torch
from torch import nn
from adv_generator import LinfPGD, inf_generator, FGSM
import tqdm
import time
from models import init_params as w_init
from expand_exp import curvature, weight_assign


def boundary_shrink(ori_model, train_forget_loader, dt, dv, test_loader, device, evaluate,
                    bound=0.1, step=8 / 255, iter=5, poison_epoch=10, forget_class=0, path='./',
                    extra_exp=None, lambda_=0.7, bias=-0.5, slope=5.0):
    start = time.time()
    norm = True  # None#True if data_name != "mnist" else False
    random_start = False  # False if attack != "pgd" else True

    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)
    start_time = time.time()
    # adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)
    adv = FGSM(test_model, bound, norm, random_start, device)
    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=0.00001, momentum=0.9)

    num_hits = 0
    num_sum = 0
    nearest_label = []

    for itr in tqdm.tqdm(range(poison_epoch * batches_per_epoch)):

        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        test_model.eval()
        x_adv = adv.perturb(x, y, target_y=None, model=test_model, device=device)
        adv_logits = test_model(x_adv)
        pred_label = torch.argmax(adv_logits, dim=1)
        if itr >= (poison_epoch - 1) * batches_per_epoch:
            nearest_label.append(pred_label.tolist())
        num_hits += (y != pred_label).float().sum()
        num_sum += y.shape[0]

        # adv_train
        unlearn_model.train()
        unlearn_model.zero_grad()
        optimizer.zero_grad()

        ori_logits = unlearn_model(x)
        ori_loss = criterion(ori_logits, pred_label)

        # loss = ori_loss  # - KL_div
        if extra_exp == 'curv':
            ori_curv = curvature(ori_model, x, y, h=0.9)[1]
            cur_curv = curvature(unlearn_model, x, y, h=0.9)[1]
            delta_curv = torch.norm(ori_curv - cur_curv, p=2)
            loss = ori_loss + lambda_ * delta_curv  # - KL_div
        elif extra_exp == 'weight_assign':
            weight = weight_assign(adv_logits, pred_label, bias=bias, slope=slope)
            ori_loss = (torch.nn.functional.cross_entropy(ori_logits, pred_label, reduction='none') * weight).mean()
            loss = ori_loss
        else:
            loss = ori_loss  # - KL_div
        loss.backward()
        optimizer.step()

    print('attack success ratio:', (num_hits / num_sum).float())
    # print(nearest_label)
    print('boundary shrink time:', (time.time() - start_time))
    # np.save('nearest_label', nearest_label)
    torch.save(unlearn_model, '{}boundary_shrink_unlearn_model.pth'.format(path))

    test_forget_loader, test_remain_loader = utils.get_forget_loader(dv, forget_class)
    _, train_remain_loader = utils.get_forget_loader(dt, forget_class)

    mode = 'pruned' if evaluate else ''
    _, test_acc = eval(model=unlearn_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
                       name='test set all class')
    _, forget_acc = eval(model=unlearn_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set forget class')
    _, remain_acc = eval(model=unlearn_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set remain class')
    _, train_forget_acc = eval(model=unlearn_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set forget class')
    _, train_remain_acc = eval(model=unlearn_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set remain class')
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
          .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    end = time.time()
    print('Time Consuming:', end - start, 'secs')
    return unlearn_model


def boundary_expanding(ori_model, train_forget_loader, test_loader, test_forget_loader, test_remain_loader,
                       train_remain_loader, optimization, device, evaluate, path='./',):
    start = time.time()

    n_filter2 = int(192 * 0.5)
    num_classes = 10
    narrow_model = copy.deepcopy(ori_model).to(device)
    feature_extrator = narrow_model.features
    classifier = narrow_model.classifier

    widen_classifier = nn.Linear(n_filter2, num_classes + 1)
    w_init(widen_classifier)
    widen_model = nn.Sequential(feature_extrator, widen_classifier)
    widen_model = widen_model.to(device)

    dict = widen_classifier.state_dict()

    for name, params in classifier.named_parameters():
        # print(name, params.data)
        if 'weight' in name:
            widen_classifier.state_dict()['weight'][0:10, ] = classifier.state_dict()[name][:, ]
        elif 'bias' in name:
            widen_classifier.state_dict()['bias'][0:10, ] = classifier.state_dict()[name][:, ]

    forget_data_gen = inf_generator(train_forget_loader)
    batches_per_epoch = len(train_forget_loader)
    finetune_epochs = 10

    criterion = loss_picker('cross')
    optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # centr_optimizer = optimizer_picker(optimization, widen_model.parameters(), lr=0.00001, momentum=0.9)
    # adv_optimizer = optimizer_picker(optimization, adv_model.parameters(), lr=0.001, momentum=0.9)

    for itr in tqdm.tqdm(range(finetune_epochs * batches_per_epoch)):
        x, y = forget_data_gen.__next__()
        x = x.to(device)
        y = y.to(device)

        widen_logits = widen_model(x)

        # target label
        target_label = torch.ones_like(y, device=device)
        target_label *= num_classes

        # adv_train
        widen_model.train()
        widen_model.zero_grad()
        optimizer.zero_grad()

        widen_loss = criterion(widen_logits,
                               target_label)

        widen_loss.backward()
        optimizer.step()

    pruned_classifier = nn.Linear(n_filter2, num_classes)
    for name, params in widen_model[1].named_parameters():
        # print(name)
        if 'weight' in name:
            pruned_classifier.state_dict()['weight'][:, ] = widen_model[1].state_dict()[name][0:10, ]
        elif 'bias' in name:
            pruned_classifier.state_dict()['bias'][:, ] = widen_model[1].state_dict()[name][0:10, ]

    pruned_model = nn.Sequential(feature_extrator, pruned_classifier)
    pruned_model = pruned_model.to(device)

    mode = 'pruned' if evaluate else ''
    _, test_acc = eval(model=pruned_model, data_loader=test_loader, mode=mode, print_perform=evaluate, device=device,
                       name='test set all class')
    _, forget_acc = eval(model=pruned_model, data_loader=test_forget_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set forget class')
    _, remain_acc = eval(model=pruned_model, data_loader=test_remain_loader, mode=mode, print_perform=evaluate,
                         device=device, name='test set remain class')
    _, train_forget_acc = eval(model=pruned_model, data_loader=train_forget_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set forget class')
    _, train_remain_acc = eval(model=pruned_model, data_loader=train_remain_loader, mode=mode, print_perform=evaluate,
                               device=device, name='train set remain class')
    print('test acc:{:.2%}, forget acc:{:.2%}, remain acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}'
          .format(test_acc, forget_acc, remain_acc, train_forget_acc, train_remain_acc))
    end = time.time()
    print('Time Consuming:', end - start, 'secs')

    torch.save(widen_model, '{}boundary_expand_widen_model.pth'.format(path))
    torch.save(pruned_model, '{}boundary_expand_pruned_model.pth'.format(path))
    return pruned_model

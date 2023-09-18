import argparse
import numpy as np
import boundary_unlearning
from utils import *
from trainer import *


def seed_torch(seed=2022):
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_torch()
    parser = argparse.ArgumentParser("Boundary Unlearning")
    parser.add_argument('--method', type=str, default='boundary_shrink',
                        choices=['boundary_shrink', 'boundary_expanding'], help='unlearning method')
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['mnist', 'cifar10'],
                        help='dataset, mnist or cifar10')
    parser.add_argument('--model_name', type=str, default='AllCNN', choices=['MNISTNet', 'AllCNN'], help='model name')
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='training epoch')
    parser.add_argument('--forget_class', type=int, default=4, help='forget class')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='checkpoints directory')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--train', action='store_true', help='Train model from scratch')
    parser.add_argument('--evaluation', action='store_true', help='evaluate unlearn model')
    parser.add_argument('--extra_exp', type=str, help='optional extra experiment for boundary shrink',
                        choices=['curv', 'weight_assign', None])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_dir(args.dataset_dir)
    create_dir(args.checkpoint_dir)
    path = args.checkpoint_dir + '/'

    trainset, testset = get_dataset(args.data_name, args.dataset_dir)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, device=device)

    forget_class = args.forget_class
    num_forget = 5000
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index \
        = get_unlearn_loader(trainset, testset, forget_class, args.batch_size, num_forget)

    if args.train:
        print('=' * 100)
        print(' ' * 25 + 'train original model and retrain model from scratch')
        print('=' * 100)
        ori_model = train_save_model(train_loader, test_loader, args.model_name, args.optim_name, args.lr,
                                     args.epoch, device, path + args.data_name + "_original_model")
        print('\noriginal model acc:\n', test(ori_model, test_loader))
        retrain_model = train_save_model(train_remain_loader, test_remain_loader, args.model_name, args.optim_name,
                                         args.lr, args.epoch, device, path + args.data_name + "_retrain_model")
        print('\nretrain model acc:\n', test(retrain_model, test_loader))
    else:
        print('=' * 100)
        print(' ' * 25 + 'load original model and retrain model')
        print('=' * 100)
        ori_model = torch.load('{}.pth'.format(path + args.data_name + "_original_model"),
                               map_location=torch.device('cpu')).to(device)
        print('\noriginal model acc:\n', test(ori_model, test_loader))
        retrain_model = torch.load('{}.pth'.format(
            path + args.data_name + "_retrain_model"), map_location=torch.device('cpu')).to(device)
        print('\nretrain model acc:\n', test(retrain_model, test_loader))

    if args.method == 'boundary_shrink':
        print('*' * 100)
        print(' ' * 25 + 'begin boundary shrink unlearning')
        if args.extra_exp:
            print(' ' * 20 + 'with extra experiment curvature regularization' if args.extra_exp == 'curv' else
                  ' ' * 20 + 'with extra experiment weight assign')
        print('*' * 100)
        unlearn_model = boundary_unlearning.boundary_shrink(ori_model, train_forget_loader, trainset, testset,
                                                            test_loader, device, args.evaluation,
                                                            forget_class=args.forget_class, path=path,
                                                            extra_exp=args.extra_exp)
    elif args.method == 'boundary_expanding':
        print('*' * 100)
        print(' ' * 25 + 'begin boundary expanding unlearning')
        print('*' * 100)
        unlearn_model = boundary_unlearning.boundary_expanding(ori_model, train_forget_loader, test_loader,
                                                               test_forget_loader, test_remain_loader,
                                                               train_remain_loader, args.optim_name, device,
                                                               args.evaluation, path=path)

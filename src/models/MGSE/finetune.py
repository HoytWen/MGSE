import os
import os.path as osp
import sys
import argparse
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from models.MGSE import MGSEConfig
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import pandas as pd
from time import time

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader

from utils.util_funcs import time2str
from models.MGSE.model import Student_Model_graphpred
from models.MGSE.gnn import GNN_graphpred
from models.MGSE.splitters import scaffold_split, random_split, random_scaffold_split
from utils.early_stopper import EarlyStopping
from utils import *
from utils.evaluation import save_results

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def train(args, model, device, loader, optimizer, scheduler, criterion):
    model.train()

    if scheduler is not None:
        scheduler.step()

    for step, data in enumerate(loader):
        data = data.to(device)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if args.pred_type == 'cat':
            pred = model(x, edge_index, edge_attr, batch)
            y = data.y.view(pred.shape).to(torch.float64)
            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(pred.double(), (y + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        elif args.pred_type == 'ensemble':
            pred_list = model(x, edge_index, edge_attr, batch)
            y = data.y.view(pred_list[0].squeeze(0).shape).to(torch.float64)
            # Whether y is non-null or not.
            is_valid = y ** 2 > 0

            loss = 0
            for i in range(args.h_level):
                # Loss matrix
                loss_mat = criterion(pred_list[i].squeeze(0).double(), (y + 1) / 2)
                # loss matrix after removing null target
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss += torch.sum(loss_mat) / torch.sum(is_valid)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            raise ValueError


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, data in enumerate(loader):
        data = data.to(device)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if args.pred_type == 'cat':
            with torch.no_grad():
                pred = model(x, edge_index, edge_attr, batch)
            y_true.append(data.y.view(pred.shape))
            y_scores.append(pred)

        elif args.pred_type == 'ensemble':
            with torch.no_grad():
                pred_list = model(x, edge_index, edge_attr, batch)
                y_true.append(data.y.view(pred_list[0].squeeze(0).shape))
                ensemble_pred = torch.mean(torch.cat(pred_list, dim=0), dim=0)
                y_scores.append(ensemble_pred)
        else:
            raise ValueError

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


@time_logger
def finetune_MGSE(args):
    # ! Init Arguments
    exp_init(args.gpus)
    cf = MGSEConfig(args)
    cf.compute_dev = torch.device("cuda:0" if args.gpus >= 0 and torch.cuda.is_available() else "cpu")
    setup_seed(args.seed)

    from models.MGSE.loader import MoleculeDataset

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    data = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
    print(args.dataset)
    cf.feat_dim = max(data.num_features, 1)
    cf.n_class = data.num_classes

    split, split_ratio = args.split.split('_')
    split_ratio = float(split_ratio)/100
    assert split_ratio > 0 and split_ratio < 1.
    frac_train = split_ratio
    frac_valid = (1-frac_train) / 2.
    frac_test = frac_valid

    if split == "scaffold":
        smiles_list = pd.read_csv('./dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()

        train_dataset, valid_dataset, test_dataset = scaffold_split(data, smiles_list, null_value=0,
                                                                    frac_train=frac_train, frac_valid=frac_valid, frac_test=frac_test)

    elif split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(data, null_value=0, frac_train=frac_train, frac_valid=frac_valid,
                                                                  frac_test=frac_test, seed=cf.split_seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('./dataset/' + cf.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(data, smiles_list, null_value=0,
                                                                           frac_train=frac_train, frac_valid=frac_valid,
                                                                           frac_test=frac_test, seed=cf.split_seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


    n_protos = [int(x) for x in cf.n_protos.split('_')]
    h_level = len(n_protos)
    args.h_level = h_level

    if cf.d_epochs > 0:
        model = Student_Model_graphpred(h_level, num_tasks, cf, args.pred_type)
        file_path = cf.student_file + f"_scp{cf.d_epochs}" + ".pth"
        if osp.exists(file_path):
            print('Load pretrained model existing pretrained file')
            checkpoint = torch.load(file_path)
            model.encoders.encoders.load_state_dict(checkpoint['encoders'])
        else:
            print(file_path)
            raise ValueError

        model.to(cf.compute_dev)

        model_param_group = []
        if args.frozen:
            model_param_group.append({"params": model.encoders.parameters(), "lr": 0})
        else:
            model_param_group.append({"params": model.encoders.parameters()})

        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.ft_lr * args.lr_scale})
        if args.pred_type == 'attention':
            model_param_group.append({"params": model.level_weight, "lr": args.ft_lr * args.lr_scale})

    else:
        model = GNN_graphpred(num_layer=cf.n_layer, emb_dim=cf.n_hidden, num_tasks=num_tasks, JK=cf.JK,
                              drop_ratio=cf.ft_dropout, pred=True)
        file_path = cf.teacher_file + f"{cf.teacher_model}" + ".pth"
        if osp.exists(file_path):
            print('Load pretrained model existing pretrained file')
            checkpoint = torch.load(file_path, map_location=cf.compute_dev)
            if cf.teacher_model == 'graphmae':
                model.gnn.load_state_dict(checkpoint)
            else:
                model.gnn.load_state_dict(checkpoint['encoder'])
        else:
            print(file_path)
            raise ValueError

        model.to(cf.compute_dev)

        model_param_group = []
        if args.frozen:
            model_param_group.append({"params": model.gnn.parameters(), "lr": 0})
        else:
            model_param_group.append({"params": model.gnn.parameters()})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.ft_lr * args.lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.ft_lr, weight_decay=0)
    print(optimizer)
    scheduler = StepLR(optimizer, step_size=30*int(cf.ft_epochs // 100), gamma=0.3)
    # scheduler = None

    # run fine-tuning
    best_valid = 0
    best_valid_test = 0
    last_epoch_test = 0
    best_epoch = 0
    training_start_time = time()
    for epoch in range(1, args.ft_epochs + 1):
        t0 = time()
        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])

        train(args, model, cf.compute_dev, train_loader, optimizer, scheduler, criterion)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, cf.compute_dev, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, cf.compute_dev, val_loader)
        test_acc = eval(args, model, cf.compute_dev, test_loader)

        if val_acc > best_valid:
            best_valid = val_acc
            best_valid_test = test_acc
            best_epoch = epoch

        if epoch == args.ft_epochs:
            last_epoch_test = test_acc

        print_log({'Epoch': epoch, 'Time': time() - t0, 'Train': train_acc,'Valid': val_acc,
                            'Test': test_acc, 'BestValid': best_valid,
                            'BestEpoch': best_epoch, 'AccordTest': best_valid_test})

    print_log({'AccordTest': best_valid_test, 'LastTest': last_epoch_test})

    cf.training_time = time2str(time() - training_start_time)
    result = {'val_acc': round(best_valid, 4), 'test_acc': round(best_valid_test, 4), 'final_acc': round(last_epoch_test, 4)}
    res = {'best_epoch': best_epoch, **result}
    save_results(cf, res)

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    parser = MGSEConfig.add_exp_setting_args(parser)
    exp_args = parser.parse_known_args()[0]
    parser = MGSEConfig(exp_args).add_model_specific_args(parser)
    args = parser.parse_args()
    finetune_MGSE(args)


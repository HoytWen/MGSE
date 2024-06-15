from time import time

import torch as th
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast

from models.MGSE.model import Student_Model, Teacher_Model, ProjectNet, Student_ProjectNet
from models.MGSE.scheduler import CosineDecayScheduler
from models.MGSE.h_cluster import run_hkmeans_faiss

from utils.util_funcs import print_log
import wandb
import math
import copy
from tqdm import tqdm

def _similarity(h1: th.Tensor, h2: th.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class Student_Trainer():
    def __init__(self, dataset, cf):
        self.__dict__.update(cf.__dict__)
        self.cf = cf
        self.res_file = cf.res_file
        self.device = cf.compute_dev
        self.batch_size = cf.batch_size
        self.JK = cf.JK
        self.n_hidden = cf.n_hidden
        self.student_tau = cf.student_tau
        self.use_scheduler = cf.use_scheduler
        self.data_len = len(dataset)
        self.lam_r = cf.lam_r
        self.lam_b = cf.lam_b
        self.lam_c = cf.lam_c
        self.tp_init = cf.tp_init
        self.sp_init = cf.sp_init
        self.teacher_proj = cf.teacher_proj
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=cf.student_tau), mode='G2G', intraview_negs=cf.intra_negative).to(cf.compute_dev)
        self.cross_align = cf.cross_align
        self.two_aug = cf.two_aug
        self.wandb = cf.wandb

        self.dataloader = DataLoader(dataset, batch_size=cf.batch_size, shuffle=True, num_workers=8)
        if self.two_aug == 1:
            self.aug1 = A.RandomChoice([A.NodeDropping(pn=0.2), A.EdgeRemoving(pe=0.2), A.RWSampling(num_seeds=1000, walk_length=10)], 1)
            self.aug2 = A.RandomChoice([A.NodeDropping(pn=0.2), A.EdgeRemoving(pe=0.2), A.RWSampling(num_seeds=1000, walk_length=10)], 1)
        elif self.two_aug == 2:
            self.aug1 = A.Identity()
            self.aug2 = A.Identity()
        else:
            self.aug1 = A.Identity()
            self.aug2 = A.RandomChoice([A.NodeDropping(pn=0.2), A.EdgeRemoving(pe=0.2), A.RWSampling(num_seeds=1000, walk_length=10)], 1)

        model_param_group = []
        self.teacher_model = Teacher_Model(teacher_model=cf.teacher_model, cf=cf).to(self.device)
        if not cf.teacher_model == "":
            self.teacher_file_path = 'pretrain/' + f"{cf.teacher_model}/{cf.teacher_model}" + ".pth"
            teacher_ckpt = th.load(self.teacher_file_path)
            if self.teacher_proj == 1:
                self.teacher_model.encoder.load_state_dict(teacher_ckpt['encoder'])
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                if cf.teacher_model == 'graphlog':
                    self.teacher_proj = ProjectNet(cf.n_hidden).to(self.device)
                    self.teacher_proj.load_state_dict(teacher_ckpt['proj'])
                else:
                    self.teacher_proj = nn.Sequential(nn.Linear(cf.n_hidden, cf.n_hidden), nn.ReLU(inplace=True), nn.Linear(cf.n_hidden, cf.n_hidden)).to(self.device)
                    self.teacher_proj.load_state_dict(teacher_ckpt['proj'])
                for p in self.teacher_proj.parameters():
                    p.requires_grad = False
            elif self.teacher_proj == 2:
                self.teacher_model.encoder.load_state_dict(teacher_ckpt)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                if cf.teacher_model == 'graphlog':
                    self.teacher_proj = ProjectNet(cf.n_hidden).to(self.device)
                else:
                    self.teacher_proj = nn.Sequential(nn.Linear(cf.n_hidden, cf.n_hidden), nn.ReLU(inplace=True), nn.Linear(cf.n_hidden, cf.n_hidden)).to(self.device)
                model_param_group += [{"params": self.teacher_proj.parameters(), "lr": self.cf.lr}]
            else:
                self.teacher_model.encoder.load_state_dict(teacher_ckpt)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
        else:
            raise ValueError

        self.teacher_prototypes = None
        if self.tp_init == 'pretrain':
            print('Load pretrained prototypes')
            checkpoint = th.load(self.teacher_file_path)
            self.teacher_prototypes = checkpoint['prototypes']
            self.h_level = len(self.teacher_prototypes)
            for l in range(len(self.teacher_prototypes)):
                self.teacher_prototypes[l].requires_grad = False
        else:
            self.n_protos = [int(x) for x in cf.n_protos.split('_')]
            cf.h_level = len(self.n_protos)
            self.h_level = cf.h_level

        if self.sp_init == 'teacher':
            print('Use teacher prototypes for student model')
            self.student_prototypes = self.teacher_prototypes
        elif self.sp_init == 'random':
            print('Initialize student prototypes randomly')
            self.student_prototypes = self.random_init(bp=True)
        elif self.sp_init == 'faiss':
            print('Initialize student prototypes with faiss')
            self.student_prototypes = self.run_cluster()
        for l in range(self.h_level):
            model_param_group += [
                {'params': self.student_prototypes[l], 'lr': cf.lr, 'LARS_exclude': True, 'WD_exclude': True,
                 'weight_decay': 0}]

        if not cf.student_file == "":
            self.student_file_path = cf.student_file
        else:
            raise

        self.student_model = Student_Model(h_level=self.h_level, cf=cf).to(self.device)
        self.student_proj = Student_ProjectNet(h_level=self.h_level, cf=cf).to(self.device)
        model_param_group += [{"params": self.student_model.parameters(), "lr": self.cf.lr},
                              {"params": self.student_proj.parameters(), "lr": self.cf.lr}]

        self.optimizer = th.optim.Adam(model_param_group, lr=cf.lr, weight_decay=cf.weight_decay)

        if (self.data_len % self.batch_size) == 0:
            self.step_per_epoch = self.data_len // self.batch_size
        else:
            self.step_per_epoch = (self.data_len // self.batch_size) + 1

        if cf.use_scheduler:
            self.scheduler = CosineDecayScheduler(max_val=cf.lr, warmup_steps=self.step_per_epoch, total_steps=cf.d_epochs*self.step_per_epoch)

    def run(self):
        for epoch in range(1, self.d_epochs+1):
            t0 = time()
            loss, dloss, rloss, bloss, closs = self.train(epoch)
            print_log({'Epoch': epoch, 'Time': time() - t0, 'Loss': loss, 'DistillLoss': dloss, 'ReguLoss': rloss, 'BootLoss': bloss, 'ContrastLoss': closs})

            if epoch % self.save_freq == 0:
                th.save({
                    'encoders': self.student_model.encoders.state_dict(),
                    'projs': self.student_proj.projs.state_dict(),
                    'protos': self.student_prototypes
                }, self.student_file_path  + f"_scp{epoch}" + ".pth")
        return self.student_model

    def train(self, epoch):
        epoch_loss = 0
        epoch_dloss = 0
        epoch_rloss = 0
        epoch_bloss = 0
        epoch_closs = 0
        self.student_model.train()
        self.student_proj.train()

        epoch_iter = tqdm(self.dataloader, desc="Iteration")
        for step, data in enumerate(epoch_iter):
            if self.use_scheduler:
                current_step = int(step+(epoch-1)*self.step_per_epoch)
                lr = self.scheduler.get(current_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = th.ones((num_nodes, 1), dtype=th.float32, device=self.device)

            x, edge_index, edge_attr, batch, id = data.x, data.edge_index, data.edge_attr, data.batch, data.id
            x1, edge_index1, edge_attr1 = self.aug1(x, edge_index, edge_attr)
            x2, edge_index2, edge_attr2 = self.aug2(x, edge_index, edge_attr)

            tg1 = self.teacher_model(x1, edge_index1, edge_attr1, batch)
            tg2 = self.teacher_model(x2, edge_index2, edge_attr2, batch)
            if self.teacher_proj != 0:
                tg1, tg2 = [self.teacher_proj(x) for x in [tg1, tg2]]

            sg1_list = self.student_model(x1, edge_index1, edge_attr1, batch)
            sg2_list = self.student_model(x2, edge_index2, edge_attr2, batch)
            sg1_list = self.student_proj(sg1_list)
            sg2_list = self.student_proj(sg2_list)

            loss, dloss, rloss, bloss, closs, teacher_cnt, student_cnt, cross_cnt = self.HD_loss(tg1, tg2, sg1_list, sg2_list)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.wandb:
                wandb.log({ **teacher_cnt, **student_cnt, **cross_cnt,
                           'Loss': loss.item(), 'DistillLoss': dloss.item(), 'ReguLoss': rloss.item(), 'BootLoss': bloss.item(), 'ContrastLoss': closs.item()})

            epoch_loss += loss.item()
            epoch_dloss += dloss.item()
            epoch_rloss += rloss.item()
            epoch_bloss += bloss.item()
            epoch_closs += closs.item()

        return epoch_loss, epoch_dloss, epoch_rloss, epoch_bloss, epoch_closs

    def HD_loss(self, tg1, tg2, s1_list, s2_list):

        dloss = 0
        rloss = 0
        closs = 0
        bloss = 0
        teacher_cnt_dict = {}
        student_cnt_dict = {}
        cross_cnt_dict = {}
        for l in range(self.h_level):
            if self.tp_init == 'student':
                with th.no_grad():
                    tp1 = _similarity(tg1, self.student_prototypes[l])
                    tp2 = _similarity(tg2, self.student_prototypes[l])
                    tp1 = F.softmax(tp1 / self.student_tau, dim=-1)
                    tp2 = F.softmax(tp2 / self.student_tau, dim=-1)
                    tp1 = self.sharpen(tp1)
                    tp2 = self.sharpen(tp2)
            else:
                tp1 = _similarity(tg1, self.teacher_prototypes[l])
                tp2 = _similarity(tg2, self.teacher_prototypes[l])
                tp1 = F.softmax(tp1 / self.student_tau, dim=-1)
                tp2 = F.softmax(tp2 / self.student_tau, dim=-1)
                tp1 = self.sharpen(tp1)
                tp2 = self.sharpen(tp2)

            sg1 = s1_list[l]
            sg2 = s2_list[l]

            if self.lam_b > 0:
                bloss += self.align_pos(sg1, sg2)

            sp1 = _similarity(sg1, self.student_prototypes[l])
            sp2 = _similarity(sg2, self.student_prototypes[l])
            sp1 = F.softmax(sp1 / self.student_tau, dim=-1)
            sp2 = F.softmax(sp2 / self.student_tau, dim=-1)

            if self.lam_r > 0:
                rloss += self.me_max(sp1, sp2)

            if self.lam_c > 0:
                closs1 = th.mean(th.sum(-sp1 * th.log(sp2), dim=-1))
                closs2 = th.mean(th.sum(-sp2 * th.log(sp1), dim=-1))
                closs += (closs1 + closs2) / (self.h_level * 2.0)

            if self.cross_align == 1:
                dloss1 = th.mean(th.sum(-tp1 * th.log(sp2), dim=-1))
                dloss2 = th.mean(th.sum(-tp2 * th.log(sp1), dim=-1))
            elif self.cross_align == 2:
                dloss1 = (th.mean(th.sum(-tp1 * th.log(sp2), dim=-1)) + th.mean(th.sum(-tp1 * th.log(sp1), dim=-1))) * 0.5
                dloss2 = (th.mean(th.sum(-tp2 * th.log(sp1), dim=-1)) + th.mean(th.sum(-tp2 * th.log(sp2), dim=-1))) * 0.5
            else:
                dloss1 = th.mean(th.sum(-tp1 * th.log(sp1), dim=-1))
                dloss2 = th.mean(th.sum(-tp2 * th.log(sp2), dim=-1))

            dloss += (dloss1 + dloss2) / (self.h_level * 2.0)

            t1_pseudo_label = th.argmax(tp1, dim=1)
            t2_pseudo_label = th.argmax(tp2, dim=1)
            teacher_uniform_cnt = (t1_pseudo_label == t2_pseudo_label).long().sum()
            teacher_cnt_dict[f'TL{l+1}'] = teacher_uniform_cnt

            s1_pseudo_label = th.argmax(sp1, dim=1)
            s2_pseudo_label = th.argmax(sp2, dim=1)
            student_uniform_cnt = (s1_pseudo_label == s2_pseudo_label).long().sum()
            student_cnt_dict[f'SL{l+1}'] = student_uniform_cnt

            cross_uniform_cnt = (t1_pseudo_label == s1_pseudo_label).long().sum()
            cross_cnt_dict[f'CL{l+1}'] = cross_uniform_cnt

        if self.lam_r > 0:
            loss = dloss + self.lam_r*rloss
        else:
            rloss = th.tensor(-1)
            loss = dloss

        if self.lam_b > 0:
            loss += self.lam_b * bloss
        else:
            bloss = th.tensor(-1)

        if self.lam_c > 0:
            loss += self.lam_c * closs
        else:
            closs = th.tensor(-1)

        return loss, dloss, rloss, bloss, closs, teacher_cnt_dict, student_cnt_dict, cross_cnt_dict

    def me_max(self, sp1, sp2):

        avg_prob1 = th.mean(sp1, dim=0)
        avg_prob2 = th.mean(sp2, dim=0)
        loss1 = math.log(float(len(avg_prob1))) - th.sum(th.log(avg_prob1 ** (-avg_prob1)))
        loss2 = math.log(float(len(avg_prob2))) - th.sum(th.log(avg_prob2 ** (-avg_prob2)))
        loss = (loss1 + loss2) * 0.5

        return loss

    def align_pos(self, sg1, sg2):

        sg1 = F.normalize(sg1, dim=-1, p=2)
        sg2 = F.normalize(sg2, dim=-1, p=2)
        loss = (2 - 2 * (sg1 * sg2).sum(dim=-1)) * 0.5

        return loss.mean() / self.h_level

    def smooth(self, s1_list, s2_list):
        sloss = 0
        for l in range(self.h_level):
            s1 = s1_list[l]
            s2 = s2_list[l]
            sp1 = _similarity(s1, self.student_prototypes[l])
            sp2 = _similarity(s2, self.student_prototypes[l])

            avg_prob1 = F.softmax(sp1, dim=-1)
            avg_prob2 = F.softmax(sp2, dim=-1)

            sloss1 = th.mean(th.sum(th.log(avg_prob1 ** (-avg_prob1)), dim=1))
            sloss2 = th.mean(th.sum(th.log(avg_prob2 ** (-avg_prob2)), dim=1))
            sloss = (sloss1 + sloss2) * 0.5

        return sloss

    @th.no_grad()
    def update_centroid(self, batch_protos):
        for l in range(self.h_level):
            temp_proto = copy.deepcopy(batch_protos[l]).detach()
            self.student_prototypes[l] = self.momentum*self.student_prototypes[l] + (1. - self.momentum) * temp_proto

    @th.no_grad()
    def compute_features(self, ):
        graph_features = th.zeros(self.data_len, self.n_hidden).to(self.device)
        for step, data in enumerate(self.dataloader):
            data = data.to(self.device)
            g = self.teacher_model(data.x, data.edge_index, data.edge_attr, data.batch)
            graph_features[data.id] = g

        return graph_features.cpu().numpy()

    @th.no_grad()
    def run_cluster(self):
        print('Clustering...')
        graph_features = self.compute_features()
        cluster_result = run_hkmeans_faiss(graph_features, self.n_protos, self.cf)
        prototypes = cluster_result['centroids']

        return prototypes

    def random_init(self, bp):
        prototypes = []
        if self.teacher_prototypes is None:
            for l in range(self.h_level):
                temp_proto = th.empty(self.n_protos[l], self.n_hidden).to(self.device)
                _sqrt_k = (1. / self.n_hidden) ** 0.5
                th.nn.init.uniform_(temp_proto, -_sqrt_k, _sqrt_k)
                if bp:
                    temp_proto = th.nn.parameter.Parameter(temp_proto)
                    temp_proto.requires_grad = True
                prototypes.append(temp_proto)
        else:
            for l in range(len(self.teacher_prototypes)):
                temp_proto = th.empty(self.teacher_prototypes[l].shape).to(self.device)
                _sqrt_k = (1. / self.teacher_prototypes[l].shape[1]) ** 0.5
                th.nn.init.uniform_(temp_proto, -_sqrt_k, _sqrt_k)
                if bp:
                    temp_proto = th.nn.parameter.Parameter(temp_proto)
                    temp_proto.requires_grad = True
                prototypes.append(temp_proto)
        return prototypes

    def sharpen(self, p, T=0.25):
        sharp_p = p ** (1. / T)
        sharp_p /= th.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

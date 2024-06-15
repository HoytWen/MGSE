import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from models.MGSE.gnn import GNN, GNN_graphpred

def normalized_mse_loss(h1, h2, reduction='mean'):

    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    loss = F.mse_loss(h1, h2, reduction=reduction)

    return loss

def normalized_l1_loss(h1, h2, reduction='mean'):

    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    loss = F.l1_loss(h1, h2, reduction=reduction)

    return loss

def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = extra_pos_mask
    if extra_neg_mask is not None:
        neg_mask = neg_mask * extra_neg_mask
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

def add_extra_mask_neg(neg_mask, extra_neg_mask=None):

    assert extra_neg_mask is not None
    neg_mask = neg_mask * extra_neg_mask

    return neg_mask

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()



class Teacher_Model(nn.Module):
    def __init__(self, teacher_model, cf):
        super(Teacher_Model, self).__init__()
        self.teahcer_model = teacher_model
        self.device = cf.compute_dev
        self.pretrain_pool = global_mean_pool
        self.n_layer = cf.n_layer
        self.n_hidden = cf.n_hidden
        self.JK = cf.JK
        self.graph_pooling = cf.graph_pooling
        self.gnn_type = cf.gnn_type

        self.encoder = GNN(num_layer=cf.n_layer, emb_dim=self.n_hidden,
                                             JK=cf.JK, drop_ratio=cf.dropout, gnn_type=cf.gnn_type)

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def from_pretrained(self, model_file, key=None):
        if key is not None:
            checkpoint = torch.load(model_file)
            self.encoder.load_state_dict(checkpoint[key])
        else:
            self.encoder.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, edge_attr, batch):
        z = self.encoder(x, edge_index, edge_attr)
        g = self.pretrain_pool(z, batch)
        return g



class Student_Model(nn.Module):
    def __init__(self, h_level, cf, pred=False):
        super(Student_Model, self).__init__()
        self.n_hidden = cf.n_hidden
        self.device = cf.compute_dev
        self.batch_size = cf.batch_size
        self.pretrain_pool = global_mean_pool
        self.h_level = h_level
        self.student_layer = cf.student_layer
        self.pred = pred

        if cf.dim_align == 1:
            assert cf.n_hidden % self.h_level == 0
            self.student_n_hidden = cf.n_hidden // self.h_level
        elif cf.dim_align == 0:
            self.student_n_hidden = cf.n_hidden
        else:
            raise ValueError

        if cf.JK == 'concat':
            project_dim = self.student_n_hidden * self.student_layer
        else:
            project_dim = self.student_n_hidden

        self.encoders = nn.ModuleList()
        for _ in range(self.h_level):
            if not self.pred:
                self.encoders.append(GNN(num_layer=cf.student_layer, emb_dim=project_dim,
                                                 JK=cf.JK, drop_ratio=cf.dropout, gnn_type=cf.gnn_type))
            else:
                self.encoders.append(GNN(num_layer=cf.student_layer, emb_dim=project_dim,
                                         JK=cf.JK, drop_ratio=cf.ft_dropout, gnn_type=cf.gnn_type))

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_attr, batch):
        g_list = []
        for l in range(self.h_level):
            z = self.encoders[l](x, edge_index, edge_attr)
            g = self.pretrain_pool(z, batch)
            g_list.append(g)
        return g_list


class Student_Model_graphpred(nn.Module):
    def __init__(self, h_level, num_tasks, cf, pred_type='cat'):
        super(Student_Model_graphpred, self).__init__()
        self.n_hidden = cf.n_hidden
        self.device = cf.compute_dev
        self.pretrain_pool = global_mean_pool
        self.h_level = h_level
        self.num_tasks = num_tasks
        self.student_layer = cf.student_layer
        self.pred_type = pred_type
        self.dropout = nn.Dropout(cf.ft_dropout)

        if cf.dim_align == 1:
            assert cf.n_hidden % self.h_level == 0
            self.student_n_hidden = cf.n_hidden // self.h_level
        elif cf.dim_align == 0:
            self.student_n_hidden = cf.n_hidden
        else:
            raise ValueError

        if cf.JK == 'concat':
            project_dim = self.student_n_hidden * self.student_layer
        else:
            project_dim = self.student_n_hidden

        self.encoders = Student_Model(h_level=self.h_level, cf=cf, pred=True)

        if self.pred_type == 'cat':
            self.graph_pred_linear = torch.nn.Linear(project_dim * self.h_level, num_tasks)
        elif self.pred_type == 'mean':
            self.graph_pred_linear = nn.ModuleList()
            for l in range(self.h_level):
                self.graph_pred_linear.append(torch.nn.Linear(project_dim, num_tasks))
        elif self.pred_type == 'ensemble':
            self.graph_pred_linear = nn.ModuleList()
            for l in range(self.h_level):
                self.graph_pred_linear.append(torch.nn.Linear(project_dim, num_tasks))
        else:
            raise ValueError

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                th.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def from_pretrained(self, model_file, key=None):
        if key is not None:
            checkpoint = torch.load(model_file)
            self.encoders.load_state_dict(checkpoint[key])
        else:
            self.encoders.load_state_dict(torch.load(model_file))

    def forward(self, x, edge_index, edge_attr, batch):
        g_list = self.encoders(x, edge_index, edge_attr, batch)
        if self.pred_type == 'cat':
            g = torch.cat(g_list, dim=-1)
            g = self.graph_pred_linear(g)
            return g
        elif self.pred_type == 'mean':
            pred_list = []
            for l in range(self.h_level):
                pred_list.append(self.graph_pred_linear[l](g_list[l]).unsqueeze(0))
            g = torch.mean(torch.cat(pred_list, dim=0), dim=0)
            return g
        elif self.pred_type == 'ensemble':
            pred_list = []
            for l in range(self.h_level):
                pred_list.append(self.graph_pred_linear[l](g_list[l]).unsqueeze(0))
            return pred_list
        else:
            raise ValueError

class ProjectNet(torch.nn.Module):
    def __init__(self, rep_dim):
        super(ProjectNet, self).__init__()
        self.rep_dim = rep_dim
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.rep_dim, self.rep_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.rep_dim, self.rep_dim)
        )

    def forward(self, x):
        x_proj = self.proj(x)

        return x_proj

class Student_ProjectNet(torch.nn.Module):
    def __init__(self, h_level, cf):
        super(Student_ProjectNet, self).__init__()
        self.rep_dim = cf.n_hidden
        self.h_level = h_level
        self.projs = nn.ModuleList()

        for i in range(self.h_level):
            self.projs.append(ProjectNet(self.rep_dim))

    def forward(self, x_list):

        x_proj_list = [self.projs[l](x_list[l]) for l in range(self.h_level)]

        return x_proj_list
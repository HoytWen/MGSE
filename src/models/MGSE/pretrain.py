import os.path as osp
import sys
import argparse
sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils import *
from models.MGSE import MGSEConfig

import warnings
warnings.filterwarnings('ignore')


@time_logger
def pretrain_MGSE(args):
    # ! Init Arguments
    exp_init(args.gpus, seed=args.seed)
    # ! Import packages
    import torch as th
    from models.MGSE.train_student import Student_Trainer
    from models.MGSE.loader import MoleculeDataset
    import wandb

    cf = MGSEConfig(args)
    cf.compute_dev = th.device("cuda:0" if args.gpus >= 0 and th.cuda.is_available() else "cpu")

    data = MoleculeDataset("dataset/" + args.prt_dataset, dataset=args.prt_dataset)
    cf.feat_dim = max(data.num_features, 1)
    cf.n_class = data.num_classes
    print(cf)

    if cf.wandb:
        exp_name = cf.student_str
        wandb.init(project=f'{cf.teacher_model}_distill', config=cf, name=exp_name)

    ## ! Train the student model
    student_file = cf.student_file + f"_scp{cf.d_epochs}" + ".pth"
    if osp.exists(student_file):
        print('Distilled student model is found')
    else:
        print('Student model is not found, start training...')
        s_trainer = Student_Trainer(dataset=data, cf=cf)
        s_trainer.run()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    parser = MGSEConfig.add_exp_setting_args(parser)
    exp_args = parser.parse_known_args()[0]
    parser = MGSEConfig(exp_args).add_model_specific_args(parser)
    args = parser.parse_args()
    pretrain_MGSE(args)

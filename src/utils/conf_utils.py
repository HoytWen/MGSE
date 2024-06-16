import utils.util_funcs as uf
from abc import abstractmethod, ABCMeta
import os
from utils.proj_settings import RES_PATH, SUM_PATH, TEMP_PATH, PRETRAIN_PATH

early_stop = 30
d_epochs = 50
ft_epochs = 100

class ModelConfig(metaclass=ABCMeta):
    def __init__(self, model):
        self.model = model
        self.exp_name = 'default'
        self.seed = 0
        self.early_stop = early_stop
        self.d_epochs = d_epochs
        self.ft_epochs = ft_epochs
        self.prt_dataset = 'zinc_standard_agent'
        self.dataset = 'bbbp'
        self.eval_freq = 1
        self.seed = 0
        self.split_seed = 0
        self.gpus = '0'  # Use GPU as default device
        self.pred_type = 'cat'
        self.birth_time = uf.get_cur_time(t_format='%m_%d-%H_%M_%S')
        self.teacher_model = 'graphlog'

        self.split = 'scaffold_80'
        # Other attributes
        self._path_list = ['checkpoint_file', 'res_file', 'student_file', 'teacher_file']
        self._ignored_settings = ['_ignored_settings', 'tqdm_on', 'verbose', '_path_list', 'logger', 'log', 'gpus',
                                'compute_dev', 'n_class', 'n_feat', 'important_paras', 'screen_name']

    def __str__(self):
        # Print all attributes including data and other path settings added to the config object.
        return str({k: v for k, v in self.model_conf.items()})

    def update_modified_conf(self, conf_dict):
        self.__dict__.update(conf_dict)
        uf.mkdir_list([getattr(self, _) for _ in self._path_list])

    def  path_init(self, additional_files=[]):
        uf.mkdir_list([getattr(self, _) for _ in (self._path_list + additional_files)])

    @property
    def f_prefix(self):
        return f"{self.teacher_model}-sseed{self.split_seed}EF{self.eval_freq}sp{self.split}-{self.student_str}-{self.finetune_str}"

    @property
    @abstractmethod
    def student_str(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    @abstractmethod
    def finetune_str(self):
        # Model config to str
        return ValueError('The model config file name must be defined')

    @property
    def model_conf(self):
        # Print the model settings only.
        return {k: v for k, v in sorted(self.__dict__.items())
                if k not in self._ignored_settings}

    @property
    @abstractmethod
    def checkpoint_file(self):
        # Model config to str
        return ValueError('The checkpoint file name must be defined')

    @property
    def student_file(self):
        return f"{PRETRAIN_PATH}{self.teacher_model}/{self.prt_dataset}/{self.student_str}"

    @property
    def teacher_file(self):
        return f"{PRETRAIN_PATH}{self.teacher_model}/"

    @property
    def res_file(self):
        return f'{RES_PATH}{self.teacher_model}/{self.dataset}/{self.f_prefix}.txt'


    def sub_conf(self, sub_conf_list):
        # Generate subconfig dict using sub_conf_list
        return {k: self.__dict__[k] for k in sub_conf_list}

    @staticmethod
    def add_exp_setting_args(parser):
        parser.add_argument("-g", '--gpus', default=0, type=int,
                            help='CUDA_VISIBLE_DEVICES, -1 for cpu-only mode.')
        parser.add_argument("-pd", "--prt_dataset", type=str, default='zinc_standard_agent')
        parser.add_argument("-d", "--dataset", type=str, default='tox21')
        parser.add_argument("-o", "--frozen", type=bool, default=False)
        parser.add_argument("-sp", "--split", type=str, default='scaffold_80')
        parser.add_argument("-l", "--lr_scale", default=1, type=float)
        parser.add_argument("-p", "--pred_type", default='cat', type=str, help='how to aggregate the feature')
        parser.add_argument("-t", "--teacher_model", default='graphcl', type=str, help='teacher model')
        parser.add_argument("-e", "--early_stop", default=early_stop, type=int)
        parser.add_argument("-ss", "--save_freq", default=1, type=int)
        parser.add_argument("-ef", "--eval_freq", default=1, type=int)
        parser.add_argument("-n", "--eval_train", default=False, type=bool)
        parser.add_argument("-v", "--verbose", default=1, type=int,
                            help='Verbose level, higher level generates more log, -1 to shut down')
        parser.add_argument("-w", "--wandb_name", default='OFF', type=str, help='Wandb logger or not.')
        parser.add_argument("--d_epochs", default=50, type=int)
        parser.add_argument("--ft_epochs", default=100, type=int)
        parser.add_argument("--ft_lr", type=float, default=0.001, help='Fintune learning rate')
        parser.add_argument("--ft_dropout", type=float, default=0.5, help='Finetune dropout')
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--split_seed", default=42, type=int, help='Seed for splitting the dataset.')
        parser.add_argument('--log_on', action="store_true", help='show log or not')
        return parser

    def __str__(self):
        return f'{self.model} config: \n{self.model_conf}'


class SimpleObject():
    """
    convert dict to a config object for better attribute acccessing
    """

    def __init__(self, conf={}):
        self.__dict__.update(conf)

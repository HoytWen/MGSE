from utils.conf_utils import ModelConfig, SimpleObject
from utils.proj_settings import TEMP_PATH, LOG_PATH

class MGSEConfig(ModelConfig):

    def __init__(self, args):
        super(MGSEConfig, self).__init__('MGSE')
        # ! Model settings
        self.prt_dataset = 'zinc_standard_agent'
        self.lr = 0.001
        self.dropout = 0.0
        self.n_hidden = 300
        self.weight_decay = 1e-5
        self.n_layer = 5
        self.batch_size = 512
        self.JK = 'last'
        self.graph_pooling = 'mean'
        self.gnn_type='gin'
        self.intra_negative = False

        ### student model config
        self.teacher_proj = 1
        self.dim_align = 0
        self.student_layer = 5
        self.lam_r = 1.0
        self.lam_c = 0.0
        self.lam_b = 0.0
        self.student_tau = 0.1
        self.use_scheduler = 1
        self.cross_align = 0
        self.two_aug = 0

        self.tp_init = 'student'
        self.sp_init = 'random'
        self.n_protos = '50_21_2'

        self.wandb = True

        ## finetune config
        self.ft_lr = 0.001
        self.ft_dropout = 0.5
        self.__dict__.update(args.__dict__)
        self.path_init()
        # self.post_process_settings()

    def add_model_specific_args(self, parser):
        parser.add_argument("--lr", type=float, default=self.lr, help='Learning rate')
        parser.add_argument("--dropout", type=float, default=self.dropout, help='Dropout ratio')
        parser.add_argument("--student_layer", type=int, default=self.student_layer, help='The layer number of student model')
        parser.add_argument("--student_tau", type=float, default=self.student_tau, help='Teacher Temperature')
        parser.add_argument("--lam_r", type=float, default=self.lam_r, help='lambda r')
        parser.add_argument("--lam_b", type=float, default=self.lam_b, help='lamda b')
        parser.add_argument("--lam_c", type=float, default=self.lam_c, help='lamda c')
        parser.add_argument("--n_protos", type=str, default=self.n_protos, help='prototype numbers')
        parser.add_argument("--teacher_proj", type=int, default=self.teacher_proj, help='whether use teacher projection')
        parser.add_argument("--cross_align", type=int, default=self.cross_align, help='whether consistent align')
        parser.add_argument("--use_scheduler", type=int, default=self.use_scheduler, help='use scheduler or not')
        parser.add_argument("--wandb", type=bool, default=self.wandb, help='use wandb to log or not')
        return parser

    @property
    def student_str(self):
        model_str = f"da{self.dim_align}_sl{self.student_layer}_gt{self.gnn_type}_gp{self.graph_pooling}_jk{self.JK}"
        training_str = f"lr{self.lr}_drop{self.dropout}_bsz{self.batch_size}_us{self.use_scheduler}_lamc{self.lam_c}_lamr{self.lam_r}_lamb{self.lam_b}"
        proto_str = f"tpi{self.tp_init}_spi{self.sp_init}_np{self.n_protos}"
        distill_str = f"stau{self.student_tau}_tpj{self.teacher_proj}_ca{self.cross_align}_ta{self.two_aug}"
        return f"{model_str}_{training_str}_{proto_str}_{distill_str}"

    @property
    def finetune_str(self):
        return f"de{self.d_epochs}_fe{self.ft_epochs}_es{self.early_stop}_fl{self.ft_lr}_fd{self.ft_dropout}_pt{self.pred_type}"

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.teacher_model}/{self.dataset}/{self.f_prefix}S{self.seed}.ckpt"

    @property
    def log_path(self):
        return f"{LOG_PATH}{self.model}/{self.dataset}/{self.f_prefix}/"

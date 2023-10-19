import argparse
import os

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    parser = argparse.ArgumentParser("Uni3D training and evaluation")

    # Model
    parser.add_argument(
        '--model', 
        default='create_uni3d', 
        type=str)

    parser.add_argument(
        "--clip-model",
        type=str,
        default="RN50",
        help="Name of the vision and text backbone to use.",
    )
    parser.add_argument(
        "--pc-model",
        type=str,
        default="RN50",
        help="Name of pointcloud backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-pc",
        default='',
        type=str,
        help="Use a pretrained CLIP model vision weights with the specified tag or file path.",
    )

    parser.add_argument(
        "--lock-pointcloud",
        default=False,
        action='store_true',
        help="Lock full pointcloud's clip tower by disabling gradients.",
    )

    # Training
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--text-lr", type=float, default=None, help="Learning rate of text encoder.")
    parser.add_argument("--visual-lr", type=float, default=None, help="Learning rate of visual encoder.")
    parser.add_argument("--point-lr", type=float, default=None, help="Learning rate of pointcloud encoder.")

    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")

    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--text-wd", type=float, default=None, help="Weight decay of text encoder.")
    parser.add_argument("--visual-wd", type=float, default=None, help="Weight decay of visual encoder.")
    parser.add_argument("--point-wd", type=float, default=None, help="Weight decay of pointcloud encoder.")

    parser.add_argument("--ld", type=float, default=1.0, help="Learning rate Layer decay.")
    parser.add_argument("--text-ld", type=float, default=None, help="Learning rate Layer decay of text encoder.")
    parser.add_argument("--visual-ld", type=float, default=None, help="Learning rate Layer decay of visual encoder.")
    parser.add_argument("--point-ld", type=float, default=None, help="Learning rate Layer decay of pointcloud encoder.")
    parser.add_argument("--patch-dropout", type=float, default=0., help="flip patch dropout.")

    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )

    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )

    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    parser.add_argument(
        "--wandb-runid",
        default=None,
        type=str,
        help="wandb runid to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )

    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--grad-accumulation-steps", type=int, default=1, help="Gradient accumulation steps; only support deepspeed now."
    )

    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--drop-rate', default=0.0, type=float)
    parser.add_argument('--drop-path-rate', default=0.0, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')

    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Default cache dir to cache model checkpoint.')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Default optimizer.')

    parser.add_argument('--enable-deepspeed', action='store_true', default=False)
    parser.add_argument('--zero-stage', type=int, default=1, help='stage of ZERO')

    parser.add_argument('--use-embed', action='store_true', default=False, help='Use embeddings for iamge and text.')
    parser.add_argument('--is-large', action='store_true', default=False, help='whether to use large minipointnet')


    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Step interval to store embeddings",
    )
    parser.add_argument(
        '--print-freq', 
        default=10, 
        type=int, 
        help='print frequency')

    # Data
    parser.add_argument('--output-dir', default='./outputs', type=str, help='output dir')
    parser.add_argument('--pretrain_dataset_name', default='shapenet', type=str)
    parser.add_argument('--pretrain_dataset_prompt', default='shapenet_64', type=str)
    parser.add_argument('--validate_dataset_name', default='modelnet40', type=str)
    parser.add_argument('--validate_dataset_name_lvis', default='objaverse_lvis', type=str)
    parser.add_argument('--validate_dataset_name_scanobjnn', default='scanobjnn_openshape', type=str)
    parser.add_argument('--validate_dataset_prompt', default='modelnet40_64', type=str)
    parser.add_argument('--openshape_setting', action='store_true', default=False, help='whether to use osaug, by default enabled with openshape.')
    parser.add_argument('--use_lvis', action='store_true', default=False, help='whether to use livs dataset.')

    # Pointcloud 
    parser.add_argument('--npoints', default=8192, type=int, help='number of points used for pre-train and test.')
    parser.add_argument('--use_height', action='store_true', default=False, help='whether to use height informatio, by default enabled with PointNeXt.')
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--group-size", type=int, default=32, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument("--embed-dim", type=int, default=512, help="teacher embedding dimension.")

    # Evaluation
    parser.add_argument('--evaluate_3d', action='store_true', help='eval 3d only')
    parser.add_argument('--ckpt_path', default='', help='the ckpt to test 3d zero shot')

    args = parser.parse_args(args)

    if args.cache_dir is not None:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir  # huggingface model dir

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            os.environ['ENV_TYPE'] = "deepspeed"
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.9.4'")
            exit(0)
    else:
        os.environ['ENV_TYPE'] = "pytorch"
        ds_init = None

    return args, ds_init

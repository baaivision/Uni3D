from collections import OrderedDict
import math
import time
import wandb

import torch.cuda.amp as amp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import collections

from data.datasets import *
# from data.datasets import customized_collate_fn

from utils import utils
from utils.utils import get_dataset
from utils.tokenizer import SimpleTokenizer
from utils.distributed import is_master, init_distributed_device, world_info_from_env, create_deepspeed_config
from utils.params import parse_args
from utils.logger import setup_logging
from utils.scheduler import warmup_cosine_lr
from utils.optim import create_optimizer, get_all_parameters, get_loss_scale_for_deepspeed, get_grad_norm_

from datetime import datetime

import open_clip
import models.uni3d as models

best_acc1 = 0

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def compute_embedding(clip_model, texts, image):
    text_embed_all = []
    for i in range(texts.shape[0]):
        text_for_one_sample = texts[i]
        text_embed = clip_model.encode_text(text_for_one_sample)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.mean(dim=0)
        text_embed_all.append(text_embed)

    texts = torch.stack(text_embed_all)
    image = clip_model.encode_image(image)
    image = image / image.norm(dim=-1, keepdim=True)
    texts = texts.clone().detach()
    image = image.clone().detach()
    return texts, image

def main(args):
    args, ds_init = parse_args(args)

    global best_acc1

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True 
   
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])
    
    if ds_init is not None:
        dsconfg_path = os.path.join(os.getcwd(), "dsconfig", args.name)
        os.makedirs(dsconfg_path, exist_ok=True)
        create_deepspeed_config(args)

    # fix the seed for reproducibility
    # random_seed(args.seed, args.rank)

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            logging.error("Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    
    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        wandb.init(project=args.wandb_project_name, 
                name=args.name,
                notes=args.wandb_notes,
                config=vars(args), 
                settings=wandb.Settings(start_method="fork"))
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    logging.info("=> create clip teacher...")
    # It is recommended to download clip model in advance and then load from the local
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name=args.clip_model, pretrained=args.pretrained) 
    clip_model.to(device)

    # create model
    logging.info("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(args=args)
    model.to(device)
    model_without_ddp = model
    
    # evaluate model
    if args.evaluate_3d:
        logging.info("=> evaluating...")
        zero_stats, zero_stats_lvis, zero_results_scanobjnn = test_zeroshot_3d(args, model, clip_model)
        logging.info(zero_stats)
        logging.info(zero_stats_lvis)
        logging.info(zero_results_scanobjnn)
        return
    
    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)

    # print number of parameters
    total_n_parameters = sum(p.numel() for p in model.parameters())
    logging.info(f'number of total params: {total_n_parameters}')
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params with requires_grad: {n_parameters}')

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    # if args.distributed and not args.horovod:
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.enable_deepspeed:
            ddp_args = {}
            if args.ddp_static_graph:
                # this doesn't exist in older PyTorch, arg only added if enabled
                ddp_args['static_graph'] = True
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
            model_without_ddp = model.module

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.pretrain_dataset_name is not None:                
        if not args.enable_deepspeed:
            scaler = amp.GradScaler() if args.precision == "amp" else None
            optimizer = create_optimizer(args, model_without_ddp)
        else:
            scaler = None

            if args.optimizer != "lamb" and args.optimizer != "adamw":
                optimizer, optimizer_params = create_optimizer(
                    args,
                    model_without_ddp,
                    return_params=True)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
            else:
                optimizer_params = get_all_parameters(args, model)
                model, optimizer, _, _ = ds_init(
                    args=args,
                    model=model,
                    model_parameters=optimizer_params,
                    dist_init_required=not args.distributed,
                )
        if is_master(args, local=args.log_local):
            logging.info(f"num of optimizer.param_groups: {len(optimizer.param_groups)}")


    # define loss function (criterion)
    criterion = models.get_filter_loss(args).to(device)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.enable_deepspeed:
            if os.path.exists(args.resume):
                import glob
                all_checkpoints = glob.glob(os.path.join(args.resume, 'epoch_*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('/')[-1].split('_')[1]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    start_epoch = latest_ckpt
                    _, client_states = model.load_checkpoint(args.resume, tag='epoch_%d' % latest_ckpt) #tag=f"epoch_{completed_epoch}"
                    # best_acc1 = checkpoint['best_acc1'] 
                    best_acc1 = client_states['best_acc1']    
                    # best_acc1 = 75.485               
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {latest_ckpt})")
                else:
                    logging.info("=> no checkpoint found at '{}'".format(args.resume))
            else:
                logging.info("=> '{}' is not existing!".format(args.resume))
        else:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume, map_location='cpu')
                if 'epoch' in checkpoint:
                    # resuming a train checkpoint w/ epoch and optimizer state
                    start_epoch = checkpoint["epoch"]
                    sd = checkpoint["state_dict"]
                    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items()}
                    model.load_state_dict(sd)
                    if optimizer is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    if scaler is not None and 'scaler' in checkpoint:
                        scaler.load_state_dict(checkpoint['scaler'])
                    logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
                    best_acc1 = checkpoint['best_acc1']
                else:
                    # loading a bare (model only) checkpoint for fine-tune or evaluation
                    model.load_state_dict(checkpoint)
                    logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                logging.info("=> no checkpoint found at '{}'".format(args.resume))

    logging.info("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = get_dataset(train_transform, tokenizer, args, 'train')
    val_dataset = get_dataset(None, tokenizer, args, 'val')
    val_dataset_lvis = get_dataset(None, tokenizer, args, 'val_lvis')
    val_dataset_scanobjnn = get_dataset(None, tokenizer, args, 'val_scanobjnn')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        val_lvis_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_lvis)
        val_scanobjnn_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset_scanobjnn)

    else:
        train_sampler = None
        val_sampler = None
        val_lvis_sampler = None
        val_scanobjnn_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=customized_collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)

    val_lvis_loader = torch.utils.data.DataLoader(
        val_dataset_lvis, batch_size=args.batch_size, shuffle=(val_lvis_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_lvis_sampler, drop_last=False)

    val_scanobjnn_loader = torch.utils.data.DataLoader(
        val_dataset_scanobjnn, batch_size=args.batch_size, shuffle=(val_scanobjnn_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_scanobjnn_sampler, drop_last=False)
    # create scheduler if train
    scheduler = None
    if optimizer is not None:
        total_steps = len(train_loader) * args.epochs
        if is_master(args):
            logging.info(f"total_steps: {total_steps}")
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

    logging.info(f"beginning training")
    best_epoch = -1

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        completed_epoch = epoch + 1

        train_stats = train(train_loader, clip_model, model, criterion, optimizer, scaler, scheduler, epoch, args)
        val_stats = {"acc1": -1}
        
        scaler_state = None if scaler is None else scaler.state_dict()

        with amp.autocast(enabled=not args.disable_amp):
            val_stats = test_zeroshot_3d_core(val_loader, args.validate_dataset_name, model, clip_model, tokenizer, args, "modelnet")
            logging.info(val_stats)
            val_lvis_stats = test_zeroshot_3d_core(val_lvis_loader, args.validate_dataset_name_lvis, model, clip_model, tokenizer, args, "lvis")
            logging.info(val_lvis_stats)
            val_scanobjnn_stats = test_zeroshot_3d_core(val_scanobjnn_loader, args.validate_dataset_name_scanobjnn, model, clip_model, tokenizer, args, 'scanobjnn')
            logging.info(val_scanobjnn_stats)

            acc1 = val_lvis_stats["acc1"]

            is_best = acc1 > best_acc1
            if is_best:
                best_epoch = epoch

            best_acc1 = max(acc1, best_acc1)

            # Saving checkpoints.
            # is_master(args) can not be here while using deepspped, otherwise ckpt can not be saved
            if args.logs and args.logs.lower() != 'none' and args.enable_deepspeed:
                deepspeed_checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
                if completed_epoch == args.epochs or (
                        args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
                    ):
                        client_state = {'epoch': completed_epoch,
                                        'best_acc1': best_acc1,}
                        model.save_checkpoint(save_dir=deepspeed_checkpoint_path, tag="epoch_%s" % str(completed_epoch), client_state=client_state)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in val_stats.items()},
                    **{f'test_lvis_{k}': v for k, v in val_lvis_stats.items()},
                    **{f'test_scanobjnn_{k}': v for k, v in val_scanobjnn_stats.items()},
                    'epoch': epoch,
                    'best_acc1': best_acc1,
                    'best_epoch': best_epoch}

        # if utils.is_main_process() and args.wandb:
        if args.wandb and is_master(args):
            wandb.log(log_stats)
            # wandb.watch(model)
            
    if args.wandb and is_master(args):
        wandb.finish()

def train(train_loader, clip_model, model, criterion, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        step = epoch * iters_per_epoch + optim_iter # global training iteration
        if not args.skip_scheduler:
            scheduler(step)

        # measure data loading time
        data_time.update(time.time() - end)

        texts = inputs[3]
        pc = inputs[4] 
        image = inputs[5]
        rgb = inputs[6]

        use_image = inputs[2].reshape(-1)

        loss_masks = use_image.float()

        feature = torch.cat((pc, rgb), dim=-1)

        if not args.use_embed:
            logging.info('=> encoding captions')  
            texts, image = compute_embedding(clip_model, texts, image)

        inputs = [feature, texts, image]

        # to device
        inputs = [tensor.to(device=args.device, non_blocking=True) for tensor in inputs]

        if args.enable_deepspeed:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs, loss_masks)
            loss = loss_dict['loss']
            loss /= args.update_freq
        

        if not math.isfinite(loss.item()):
            logging.info(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)

            if (data_iter + 1) % args.update_freq != 0:
                continue

        # compute gradient and do SGD step
            scaler.step(optimizer)
            scaler.update()
            # model.zero_grad(set_to_none=True)
        
        elif args.enable_deepspeed:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()  
                  

        # clamp logit scale to [0, 100]

        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if args.enable_deepspeed:
                loss_scale, grad_nrom = get_loss_scale_for_deepspeed(model)
            elif scaler is not None:
                loss_scale = scaler.get_scale()
                grad_nrom = get_grad_norm_(model.parameters())
            else:
                loss_scale = 0.0
                grad_nrom = get_grad_norm_(model.parameters())

            if args.wandb and is_master(args):
                wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': loss_scale,
                        'grad_norm': grad_nrom,
                        'logit': logit_scale})
            progress.display(optim_iter)
            # break

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[-1]['lr'],
            'logit_scale': logit_scale}

def test_zeroshot_3d_core(test_loader, validate_dataset_name, model, clip_model, tokenizer, args=None, test_data=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f') 
    top3 = AverageMeter('Acc@3', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, top1, top3, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with open(os.path.join("./data", 'templates.json')) as f:
        templates = json.load(f)[args.validate_dataset_prompt]

    with open(os.path.join("./data", 'labels.json')) as f:
        labels = json.load(f)[validate_dataset_name]

    with torch.no_grad():
        logging.info('=> encoding captions')               
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).to(device=args.device, non_blocking=True)
            if len(texts.shape) < 2:
                texts = texts[None, ...]
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        per_class_stats = collections.defaultdict(int)
        per_class_correct_top1 = collections.defaultdict(int)
        per_class_correct_top3 = collections.defaultdict(int)
        per_class_correct_top5 = collections.defaultdict(int)

        for i, (pc, target, target_name, rgb) in enumerate(test_loader):
            for name in target_name:
                per_class_stats[name] += 1

            pc = pc.to(device=args.device, non_blocking=True)
            rgb = rgb.to(device=args.device, non_blocking=True)
            feature = torch.cat((pc, rgb),dim=-1)
            target = target.to(device=args.device, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(feature)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_pc = pc_features.float() @ text_features.float().t()

            # measure accuracy and record loss
            (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
            # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
            acc1, acc3, acc5 = utils.scaled_all_reduce([acc1, acc3, acc5])
            top1.update(acc1.item(), pc.size(0))
            top3.update(acc3.item(), pc.size(0))
            top5.update(acc5.item(), pc.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            top1_accurate = correct[:1].squeeze()
            top3_accurate = correct[:3].float().sum(0, keepdim=True).squeeze()
            top5_accurate = correct[:5].float().sum(0, keepdim=True).squeeze()
            for idx, name in enumerate(target_name):
                if top1_accurate[idx].item():
                    per_class_correct_top1[name] += 1
                if top3_accurate[idx].item():
                    per_class_correct_top3[name] += 1
                if top5_accurate[idx].item():
                    per_class_correct_top5[name] += 1

            if i % args.print_freq == 0:
                progress.display(i)

        top1_accuracy_per_class = {}
        top3_accuracy_per_class = {}
        top5_accuracy_per_class = {}
        for name in per_class_stats.keys():
            top1_accuracy_per_class[name] = per_class_correct_top1[name] / per_class_stats[name]
            top3_accuracy_per_class[name] = per_class_correct_top3[name] / per_class_stats[name]
            top5_accuracy_per_class[name] = per_class_correct_top5[name] / per_class_stats[name]

        top1_accuracy_per_class = collections.OrderedDict(top1_accuracy_per_class)
        top3_accuracy_per_class = collections.OrderedDict(top3_accuracy_per_class)
        top5_accuracy_per_class = collections.OrderedDict(top5_accuracy_per_class)
        logging.info(','.join(top1_accuracy_per_class.keys()))
        logging.info(','.join([str(value) for value in top1_accuracy_per_class.values()]))
        logging.info(','.join([str(value) for value in top3_accuracy_per_class.values()]))        
        logging.info(','.join([str(value) for value in top5_accuracy_per_class.values()]))
    progress.synchronize()
    logging.info('0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return {'acc1': top1.avg, 'acc3': top3.avg, 'acc5': top5.avg}

def test_zeroshot_3d(args, model, clip_model):
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    logging.info('loaded checkpoint {}'.format(args.ckpt_path))
    sd = checkpoint['module']
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd)

    tokenizer = SimpleTokenizer()

    test_dataset = utils.get_dataset(None, tokenizer, args, 'val')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )
    test_lvis_dataset = utils.get_dataset(None, tokenizer, args, 'val_lvis')
    test_lvis_loader = torch.utils.data.DataLoader(
        test_lvis_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    test_dataset_scanonjnn = utils.get_dataset(None, tokenizer, args, 'val_scanobjnn')
    test_loader_scanonjnn = torch.utils.data.DataLoader(
        test_dataset_scanonjnn, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False
    )

    results_mnet = test_zeroshot_3d_core(test_loader, args.validate_dataset_name, model, clip_model, tokenizer, args, 'modelnet')
    results_lvis = test_zeroshot_3d_core(test_lvis_loader, args.validate_dataset_name_lvis, model, clip_model, tokenizer, args, 'lvis')
    results_scanobjnn = test_zeroshot_3d_core(test_loader_scanonjnn, args.validate_dataset_name_scanobjnn, model, clip_model, tokenizer, args, 'scanobjnn')
    return results_mnet, results_lvis, results_scanobjnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct


if __name__ == '__main__':
    main(sys.argv[1:])
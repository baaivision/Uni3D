import torch
import logging
import re
import json

from .distributed import is_master

try:
    from apex.optimizers import FusedAdam
except:
    print("Please install lastest apex to use FusedAdam and FusedLAMB")
    FusedAdam, FusedLAMB = None, None

def get_num_layer_for_transformer(param_name, num_max_layer):
    layer_0 = {
        "patch_embed", 
        "pos_embed", 
        "cls_token", 
        "mask_token", 
        "conv1",
        "positional_embedding",
        "token_embedding",
        "transformer.embeddings.word_embeddings",
        "transformer.embeddings.position_embeddings",
        "transformer.embeddings.token_type_embeddings",
    }

    if any(l in param_name for l in layer_0):
        return 0

    block_regex = re.compile(r"blocks\.([0-9]+)\.")
    match_block = block_regex.search(param_name)

    #huggingface->text.transformer.encoder.layer
    layer_regex = re.compile(r"layer\.([0-9]+)\.") 
    match_layer = layer_regex.search(param_name)
    if match_block is not None:
        return int(match_block.group(1)) + 1
    elif match_layer is not None:
        return int(match_layer.group(1)) + 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_transformer(var_name, len(self.values))

def get_parameters(args, model, assigner, tower):
    filter_parameters = []
    skip = set()
    if tower == 'visual':
        lr = args.visual_lr if args.visual_lr is not None else args.lr
        weight_decay = args.visual_wd if args.visual_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'visual.' in name and 'point_encoder.' not in name]
        if hasattr(model, 'visual'):
            if hasattr(model.visual, 'no_weight_decay'):
                skip = set.union(skip, model.visual.no_weight_decay())
        skip = ['visual.' + n for n in skip]
    elif tower == 'text':
        lr = args.text_lr if args.text_lr is not None else args.lr
        weight_decay = args.text_wd if args.text_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'text.' in name]
        if hasattr(model, 'text'):
            if hasattr(model.text, 'no_weight_decay'):
                skip = set.union(skip, model.text.no_weight_decay())
        skip = ['text.' + n for n in skip]
    elif tower == 'point':
        lr = args.point_lr if args.point_lr is not None else args.lr
        weight_decay = args.point_wd if args.point_wd is not None else args.wd
        filter_parameters = [[name, param] for name, param in model.named_parameters() if 'point_encoder.visual' in name]
        if hasattr(model, 'point_encoder'):
            if hasattr(model.point_encoder.visual, 'no_weight_decay'):
                # skip = set.union(skip, model.point_encoder.visual.no_weight_decay())
                skit =  set.union(skip, {'pos_embed', 'cls_token'})
        skip = ['point_encoder.visual.' + n for n in skip]
    else:
        lr = args.lr
        weight_decay = args.wd
        exclude = lambda n: 'visual.' not in n and 'text.' not in n and 'point_encoder.visual.' not in n
        filter_parameters = [[n, p] for n, p in model.named_parameters() if exclude(n)]
        if hasattr(model, 'no_weight_decay'):
            skip = set.union(skip, model.no_weight_decay())

    get_num_layer  = assigner.get_layer_id if assigner is not None else None
    get_layer_scale = assigner.get_scale if assigner is not None else None


    parameter_group_names = {}
    parameter_group_vars = {}
    for name, param in filter_parameters:
        if not param.requires_grad:
            continue

        # if param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or 'logit_scale' in name or name in skip:
        if param.ndim <= 1 or name.endswith(".bias") or name in skip:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = tower + "_" + "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr
            }
            parameter_group_vars[group_name] = {
                "group": tower,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                "lr": lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    if is_master(args, local=args.log_local):
        logging.info(f"Tower = {tower}")
        logging.info(f"Skip weight decay name marked in tower-{tower}: {skip}")
        logging.info(f"Num of parameters group in tower-{tower}: {len(parameter_group_vars.values())}")
        logging.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
    return list(parameter_group_vars.values())


def get_assigner(args, model):
    visual_ld = args.visual_ld if args.visual_ld else args.ld
    text_ld = args.text_ld if args.text_ld else args.ld
    point_ld = args.point_ld if args.point_ld else args.ld

    
    if visual_ld < 1.0:
        visual_num_layers = model.visual.get_num_layers()
        assigner_visual = LayerDecayValueAssigner(list(visual_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
    else:
        assigner_visual = None

    if text_ld < 1.0:
        text_num_layers = model.text.get_num_layers()
        assigner_text = LayerDecayValueAssigner(list(text_ld ** (text_num_layers + 1 - i) for i in range(text_num_layers + 2)))
    else:
        assigner_text = None

    if point_ld < 1.0:
        visual_num_layers =  len(model.point_encoder.visual.blocks)
        assigner_point = LayerDecayValueAssigner(list(point_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))
    else:
        visual_num_layers = len(model.point_encoder.visual.blocks)
        assigner_point = LayerDecayValueAssigner(list(point_ld ** (visual_num_layers + 1 - i) for i in range(visual_num_layers + 2)))

    if assigner_visual is not None:
        logging.info("Assigned visual values = %s" % str(assigner_visual.values))
    if assigner_text is not None:
        logging.info("Assigned text values = %s" % str(assigner_text.values))
    if assigner_point is not None:
        logging.info("Assigned point values = %s" % str(assigner_point.values))
    return assigner_visual, assigner_text, assigner_point

def get_all_parameters(args, model):
    assigner_visual, assigner_text, assiner_point = get_assigner(args, model)
        
    parameters = []
    visual_parameters = get_parameters(args, model, assigner_visual, 'visual')
    text_parameters = get_parameters(args, model, assigner_text, 'text')
    point_parameters = get_parameters(args, model, assiner_point, 'point')
    other_parameters = get_parameters(args, model, None, 'other')

    parameters.extend(visual_parameters)
    parameters.extend(text_parameters)
    parameters.extend(point_parameters)
    parameters.extend(other_parameters)

    if len(parameters) == 0:
        parameters = model.parameters()
    return parameters

def create_optimizer(args, model, return_params=False):
    optimizer_args = dict(
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer != 'lion':
        optimizer_args['eps'] = args.eps
        
    if args.optimizer == 'fused_adam':
        base_optimizer = FusedAdam
    else:
        base_optimizer = torch.optim.AdamW

    parameters = get_all_parameters(args, model)

    optimizer = base_optimizer(parameters, **optimizer_args)

    if is_master(args, local=args.log_local):
        logging.info(f'Optimizer: {args.optimizer}')
        logging.info(f'Optimizer config: {optimizer_args}')

    if return_params:
        return optimizer, parameters
    return optimizer

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)
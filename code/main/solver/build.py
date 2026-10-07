import torch

from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer_for_retrieve_model(args, model, logger):
    params = []
    normal_params = []
    bias_params = []
    rand_init_params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        if "cross" in key:
            rand_init_params.append(value)
        elif "bias" in key:
            bias_params.append(value)
        elif "classifier" in key or "mlm_head" in key:
            rand_init_params.append(value)
        else:
            normal_params.append(value)

    params += [{"params": normal_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    params += [{"params": bias_params, "lr": args.lr * args.bias_lr_factor, "weight_decay": args.weight_decay_bias}]
    params += [{"params": rand_init_params, "lr": args.lr * args.lr_factor, "weight_decay": args.weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-7,
        )
    else:
        NotImplementedError

    return optimizer


def build_optimizer_for_selector_model(args, model, logger):
    params = []
    normal_params = []
    bias_params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        assert "selector" in key
        if "bias" in key:
            bias_params.append(value)
        else:
            normal_params.append(value)

    params += [{"params": normal_params, "lr": args.lr, "weight_decay": args.weight_decay}]
    params += [{"params": bias_params, "lr": args.lr * args.bias_lr_factor, "weight_decay": args.weight_decay_bias}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-7,
        )
    else:
        NotImplementedError

    return optimizer


def build_optimizer(args, model, logger):
    if args.stage == 'train_retriever':
        optimizer = build_optimizer_for_retrieve_model(args, model, logger)
    elif args.stage == 'warmup_selector':
        optimizer = build_optimizer_for_selector_model(args, model, logger)
    elif args.stage == 'prepare_data':
        optimizer = None
    else:
        raise NotImplementedError
    return optimizer


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )

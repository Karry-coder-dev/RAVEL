import argparse
def get_args_parser():
    parser = argparse.ArgumentParser(description='Training Config')

    # config file path
    parser.add_argument('--config_file', type=str, default=None)

    #
    parser.add_argument('--stage', type=str, default='train_question',
                        choices=['train_question', 'train_answer', 'train_retriever', 'eval'])
    parser.add_argument('--interact_round', type=int, default=4)
    parser.add_argument('--num_candidates', type=int, default=10)
    parser.add_argument('--max_answer_length', type=int, default=256)
    parser.add_argument('--max_question_length', type=int, default=384)

    # backbone settings
    parser.add_argument('--pretrain_choice', default='ViT-B/16')  # whether use pretrained model
    parser.add_argument('--temperature', type=float, default=0.02,
                        help='''initial temperature value, if 0, don't use temperature''')
    parser.add_argument('--img_aug', default=False, action='store_true')

    ##vison trainsformer settings
    parser.add_argument('--img_size', type=tuple, default=(384, 128))
    parser.add_argument('--stride_size', type=int, default=16)

    # text transformer settings
    parser.add_argument('--max_retrieve_length', type=int, default=77)
    parser.add_argument('--vocab_size', type=int, default=49408)

    # solver
    parser.add_argument('--optimizer', type=str, default='Adam', help='[SGD, Adam, Adamw]')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument('--bias_lr_factor', type=float, default=2.)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-5)
    parser.add_argument('--weight_decay_bias', type=float, default=0.)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.999)

    # scheduler
    parser.add_argument('--num_epoch', type=int, default=60)
    parser.add_argument('--milestones', type=int, nargs='+', default=(20, 50))
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup_factor', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_method', type=str, default='linear')
    parser.add_argument('--lrscheduler', type=str, default='cosine')
    parser.add_argument('--target_lr', type=float, default=0)
    parser.add_argument('--power', type=float, default=0.9)

    # dataset
    parser.add_argument('--dataset_name', default='CUHK-PEDES', help='[CUHK-PEDES, ICFG-PEDES, RSTPReid]')
    parser.add_argument('--sampler', default='random', help='choose sampler from [idtentity, random]')
    parser.add_argument('--num_instance', type=int, default=4)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--test', dest='training', default=True, action='store_false')

    # evaluation setting
    parser.add_argument('--eval_period', default=1)
    parser.add_argument('--val_dataset', default='test')  # use val set when evaluate, if test use test set
    parser.add_argument('--print_freq', default=50)

    # distributed training options
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--run_name', type=str, default='OBJ_005Base')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of sample_all_gather processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up sample_all_gather training')
    return parser
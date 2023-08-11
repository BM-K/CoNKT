import os
import torch
import random
import logging
import numpy as np
import torch.distributed as dist

from argparse import ArgumentParser

class Arguments():

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--opt_level', type=str, default='O1')
        self.add_argument('--fp16', type=str, default='True')
        self.add_argument('--train', type=str, default='True')
        self.add_argument('--test', type=str, default='True')
        self.add_argument('--warmup_stage', type=str, default='True')
        self.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def add_hyper_parameters(self):
        self.add_argument('--model_type', type=str, default='t5')
        self.add_argument('--model_name_or_path', type=str, default='paust/pko-t5-small')
        self.add_argument('--patient', type=int, default=5)
        self.add_argument('--dropout', type=int, default=0.1)
        self.add_argument('--max_len', type=int, default=256)
        self.add_argument('--batch_size', type=int, default=32)
        self.add_argument('--epochs', type=int, default=10)
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--lr', type=float, default=0.00003)
        self.add_argument('--warmup_ratio', type=float, default=0.1)
        self.add_argument('--max_length', type=int, default=128)
        self.add_argument('--min_length', type=int, default=0)
        self.add_argument('--no_repeat_ngram', type=int, default=4)
        self.add_argument('--length_penalty', type=float, default=0.8)
        self.add_argument('--early_stopping', type=str, default=True)
        self.add_argument('--num_beams', type=int, default=4)
        self.add_argument('--top_p', type=float, default=0.92)
        self.add_argument('--sample_num_beams', type=int, default=4)
        self.add_argument('--max_sample_num', type=int, default=16)
        self.add_argument('--alpha', type=float, default=0.5)
    
    def add_distributed_option(self):
        self.add_argument('--rank', type=int, default=0)
        self.add_argument('--local_rank', type=int)
        self.add_argument('--num_workers', type=int, default=8)
        self.add_argument('--world_size', type=int, default=2)
        self.add_argument('--gpu_ids', nargs='+', default=['0', '1'])

    def add_data_parameters(self):
        self.add_argument('--train_data', type=str, default='train.tsv')
        self.add_argument('--test_data', type=str, default='test.tsv')
        self.add_argument('--valid_data', type=str, default='valid.tsv')
        self.add_argument('--path_to_data', type=str, default='./data/')
        self.add_argument('--path_to_save', type=str, default='./output/')
        self.add_argument('--ckpt', type=str, default='best_ckpt.pt')
        self.add_argument('--warmup_ckpt', type=str, default='warmup_ckpt_newst5.pt')
        self.add_argument('--warmup_init_ckpt', type=str, default='warmup_init_ckpt_newst5.pt')

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:print("\t", key, ":", value, "\n}")
            else:print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        if args.local_rank == 0: self.print_args(args)
        return args

class RankFilter(logging.Filter):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def filter(self, record):
        if self.args.local_rank==0:
            return record

class Setting():

    def set_logger(self, args):

        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        rank_filter = RankFilter(args)
        stream_handler.addFilter(rank_filter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.DEBUG)

        return _logger

    def set_seed(self, args):

        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def init_for_distributed(self, args):
        
        args.num_workers = len(args.gpu_ids) * 4
        args.world_size = len(args.gpu_ids)
    
        local_gpu_id = int(args.gpu_ids[args.rank])
        args.rank = int(os.environ['RANK'])
        
        torch.cuda.set_device(local_gpu_id)
        dist.init_process_group(backend='nccl',      
                                init_method='env://',
                                world_size=args.world_size,
                                rank=args.rank,)
        dist.barrier()
        
    def run(self):

        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_distributed_option()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set_logger(args)
        
        self.set_seed(args)

        self.init_for_distributed(args)

        return args, logger

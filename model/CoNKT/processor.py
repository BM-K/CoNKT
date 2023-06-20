import logging
import datetime
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torch.distributed as dist

from tqdm import tqdm
from model.utils import Metric
from data.dataloader import get_loader
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup
from model.CoNKT.net import ContrastiveNeuralTextGeneration
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}
        
        if args.warmup_stage == 'True':
            self.sorted_path = args.path_to_save + args.warmup_ckpt
        else:
            self.warmup_path = args.path_to_save + args.warmup_init_ckpt
            self.sorted_path = args.path_to_save + args.ckpt

    def run(self, inputs, mode=None):
        loss = self.config['model'](inputs, mode)
        
        return loss

    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().type(torch.FloatTensor).numpy() / self.model_progress['iter']
        
        return loss

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
    
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.args.lr)
        
        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
                    optim,
                    num_warmup_steps=self.args.warmup_ratio*train_total,
                    num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=1)

        if self.args.train == 'True' and self.args.test == 'False':
            accelerator = Accelerator(mixed_precision='bf16', 
                                      deepspeed_plugin=deepspeed_plugin,)
        else:
            accelerator = Accelerator(mixed_precision='no')

        loader, tokenizer, sampler = get_loader(self.args, self.metric)
        
        model = ContrastiveNeuralTextGeneration(self.args, tokenizer)
        model.to(self.args.device)
        
        if self.args.warmup_stage == 'False':
            model.generator.load_state_dict(torch.load(self.warmup_path))
        
        if self.args.train == 'True' and self.args.test == 'False':
            model = DDP(module=model,
                        device_ids=[self.args.rank])
        
        criterion, optimizer = self.get_object(tokenizer, model)
        
        if self.args.test == 'False':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'accelerator': accelerator,
                  'train_dl': None,
                  'valid_dl': None,
                  'test_dl': None,
                  'sampler': sampler,
                  'args': self.args,
                  'model': model}
        
        if self.args.train == 'True':
            config['model'], config['optimizer'], config['train_dl'], config['valid_dl'] = accelerator.prepare(
                    model,
                    optimizer,
                    config['loader']['train'],
                    config['loader']['valid'])
        else:
            config['model'], config['optimizer'], config['test_dl'] = accelerator.prepare(
                    model, 
                    optimizer, 
                    config['loader']['test'])

        self.config = config

        return self.config

    def train(self, epoch):
        self.config['model'].train()
        self.config['sampler']['train'].set_epoch(epoch)
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)
         
        for step, inputs in enumerate(tqdm(self.config['train_dl'])):
            self.config['optimizer'].zero_grad()
        
            loss = self.run(inputs, mode='train')
            self.config['accelerator'].backward(loss)

            self.config['optimizer'].step()
            self.config['scheduler'].step()

            self.progress(loss.data)
        
        return self.return_value()

    def valid(self, epoch):
        self.config['model'].eval()
        self.config['sampler']['valid'].set_epoch(epoch)
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(self.config['valid_dl'])):
                loss = self.run(inputs, mode='valid')
                
                if self.args.local_rank == 0:
                    self.progress(loss.data)

        if self.args.local_rank == 0:
            return self.return_value()

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.sorted_path))
        self.config['model'].eval()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.config['test_dl'])):

                inputs = batch
                self.metric.generation(self.config, inputs)
        
        return self.metric.avg_rouge()

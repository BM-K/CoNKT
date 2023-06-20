import re
import os
import csv
import torch
import logging
import torch.distributed as dist

from rouge import Rouge
from tensorboardX import SummaryWriter
from pecab import PeCab

pecab = PeCab()
logger = logging.getLogger(__name__)
writer = SummaryWriter()

class Metric():

    def __init__(self, args):
        self.args = args
        self.step = 0
        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}
        self.gen_data = list()

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def save_config(self, cp):
        config = "Config>>\n"
        for idx, (key, value) in enumerate(self.args.__dict__.items()):
            cur_kv = str(key) + ': ' + str(value) + '\n'
            config += cur_kv
        config += 'Epoch: ' + str(cp["ep"]) + '\t' + 'Valid loss: ' + str(cp['vl']) + '\n'

        with open(self.args.path_to_save+'config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        if config['args'].warmup_stage == 'True':
            warmup_nll_path = config['args'].path_to_save + config['args'].warmup_ckpt
            warmup_init_path = config['args'].path_to_save + config['args'].warmup_init_ckpt
        else:
            sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            if self.args.warmup_stage == 'True':
                torch.save(config['model'].module.module.generator.state_dict(), warmup_init_path)
                torch.save(config['model'].module.state_dict(), warmup_nll_path)
            else:
                torch.save(config['model'].module.state_dict(), sorted_path)

            self.save_config(cp)
            print(f'\n\t## SAVE valid_loss: {cp["vl"]:.4f} ##')
        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True
                writer.close()

        self.performance_check(cp)

    def result_file(self,):
        sorted_path = self.args.path_to_save + 'result.tsv'
        with open(sorted_path, 'w', encoding='utf-8') as f:
            tw = csv.writer(f, delimiter='\t')
            for data in self.gen_data: 
                tw.writerow(data)

    def rouge_score(self, config, hyp, ref):
        ref = ' '.join(pecab.morphs(ref.lower().strip()))
        hyp = ' '.join(pecab.morphs(hyp.lower().strip()))
        
        score = self.rouge.get_scores(hyp, ref)[0]
        
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] += score[metric][key]
        
        self.step += 1

    def avg_rouge(self):
        self.result_file()
        
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] /= self.step
    
        return self.rouge_scores

    def generation(self, config, inputs):
        outputs = config['model'](inputs, mode='test')

        for step, beam in enumerate(outputs):
            ref = config['tokenizer'].decode(inputs['decoder_input_ids'][step], skip_special_tokens=True)
            hyp = config['tokenizer'].decode(beam, skip_special_tokens=True)
            
            self.gen_data.append([ref, hyp])
            self.rouge_score(config, hyp, ref)

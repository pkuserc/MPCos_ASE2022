#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
import os
import json
import torch
torch.manual_seed(1)
import pickle
from preprocess.vocab import Vocab
import transformers

import json
from models.model import TransformerForConditionalGeneration


from datasets import load_metric
rouge = load_metric("rouge.py")
bleu = load_metric("bleu.py")


# ## super parameters and load vocab

# In[2]:


model_name = 'MPCOS_10'
device = torch.device('cuda:2')

load_model_from = '/tf/code_summarization/baselines/baseline_pretrained'
load_config_from = '/tf/code_summarization/baselines/baseline_pretrained'

num_task=5
batch_size = 16
code_max_len = 150
comment_max_len = 50

PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3
special_token_ids = {PAD_ID, UNK_ID, SOS_ID, EOS_ID}


# In[3]:


from models.model import TransformerConfig
config = TransformerConfig.from_pretrained(load_config_from)
config.pad_token_id=PAD_ID
config.bos_token_id=SOS_ID
config.eos_token_id=EOS_ID
config.max_length = config.comment_max_len = comment_max_len
config.code_max_len = code_max_len
config.model_name = model_name

config.prefix_seqlen = 10
config.prefix_dropout = 0.0
config.prefix_mid_dim = 800

config.use_prefix = True
config.shared_prefix_embedding = True
config.use_adapter = False
config.adapter_initializer_range=1e-2
config.use_cache = False

config.d_model = 1024
config.learning_rate = 5e-4


# In[4]:


import pickle
code_vocab = pickle.load(open('/tf/code_summarization/preprocessed_data/astattendgru/v0/code_vocab.pkl', 'rb'))
comment_vocab = pickle.load(open('/tf/code_summarization/preprocessed_data/astattendgru/v0/comment_vocab.pkl', 'rb'))
code2id = code_vocab.token2index
comment2id = comment_vocab.token2index
config.encoder_vocab_size=len(code2id)
config.decoder_vocab_size=len(comment2id)
id2comment = {v:k for k,v in comment_vocab.token2index.items()}

def preprocess(sequences, vocab, max_len):
    padded = []
    for seq in sequences:
        seq = [SOS_ID] + [vocab[i] for i in seq] + [EOS_ID]
        seq = seq[:max_len]
        seq = seq + ([PAD_ID] * (max_len - len(seq)))
        padded.append(seq)
    return {i:v for i,v in enumerate(padded)}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.ids = None
        for feature in features.values():
            ids = set(feature.keys())
            if self.ids is None:
                self.ids = ids
            else:
                self.ids = self.ids & ids
                
        self.ids = [int(i) for i in self.ids]
        self.features = {'id': self.ids}
        for k, feature in features.items():
            self.features[k] = [feature[_id] for _id in self.ids]

    def __getitem__(self, idx):
        _id = torch.tensor(self.ids[idx])
        input_ids = torch.tensor(self.features['code'][idx])
        attention_mask = (input_ids != PAD_ID).long()
        decoder_input_ids = torch.tensor(self.features['comment'][idx])
        labels =  torch.tensor(self.features['comment'][idx][1:] + [PAD_ID])
        decoder_attention_mask = (decoder_input_ids != PAD_ID).long()
        return {
            'id': _id,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.ids)


import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import learn2learn as l2l
   
class MetaTrainer(object):
    def __init__(self, config, id2comment, special_token_ids, device, meta_datasets):
        self.config = config
        self.device = device
        self.id2comment = id2comment
        self.special_token_ids = special_token_ids
        self.num_worker = 8
        self.val_metric = 'rouge2'
        self.meta_datasets = meta_datasets
        
        self.meta_dataloaders = {}
        for project, dataset in self.meta_datasets.items():
            self.meta_dataloaders[project] = {
                'support': DataLoader(dataset['support'], batch_size=batch_size, shuffle=True),
                'query': DataLoader(dataset['query'], batch_size=batch_size, shuffle=True),
                'test': DataLoader(dataset['test'], batch_size=batch_size, shuffle=False),
            }
        
    def get_input_data(self, batch, is_eval=False):
        if not is_eval:
            return {
                'input_ids': batch['input_ids'].to(self.model.device), 
                'attention_mask': batch['attention_mask'].to(self.model.device), 
                'decoder_input_ids': batch['decoder_input_ids'].to(self.model.device), 
                'decoder_attention_mask': batch['decoder_attention_mask'].to(self.model.device), 
                'labels': batch['labels'].to(self.model.device), 
            }
        else:
            return {
                'input_ids': batch['input_ids'].to(self.model.device), 
                'attention_mask': batch['attention_mask'].to(self.model.device), 
            }
    
    def ids_to_clean_text(self, generated_ids):
        if type(generated_ids) == torch.Tensor:
            if len(generated_ids.size()) >= 2:
                return [self.ids_to_clean_text(i) for i in generated_ids]
            generated_ids = generated_ids.cpu().numpy().tolist()
        return [self.id2comment[i] for i in generated_ids if i not in self.special_token_ids]

    def run_batch(self, model, batch, is_eval=False):
        if not is_eval:
            labels = batch['labels'].to(device)
            outputs = model(**self.get_input_data(batch, is_eval))
        else:
            labels = None
            if 'labels' in batch:
                labels = batch['labels']
            outputs = model.generate(**self.get_input_data(batch, is_eval))
        return outputs, labels
    
    def calc_generative_metrics(self, preds, targets):
        metrics = {k:v.mid.fmeasure for k,v in rouge.compute(predictions=[' '.join(i) for i in preds], references=[' '.join(i) for i in targets]).items()}
        metrics['bleu'] = bleu.compute(predictions=preds, references=[[i] for i in targets])['bleu']
        return metrics
    
    def evaluate(self, model, data_loader, log=False):
        y_pred = []
        y_true = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader):
                outputs, labels = self.run_batch(model, batch, is_eval=True)
                preds = self.ids_to_clean_text(outputs.cpu())
                targets = self.ids_to_clean_text(labels.cpu())
                
                y_pred.extend(preds)
                y_true.extend(targets)
        
        metrics = self.calc_generative_metrics(y_pred, y_true)
        return metrics[self.val_metric], metrics
        
    def meta_train(self, model, target_project, source_projects, save_dir,
              train_steps=12000, inner_train_steps=4, 
              valid_steps=200, inner_valid_steps=4, 
              valid_every=2, eval_start=0, early_stop=50):
        
        self.model = model
        best_bleu4 = 0
        best_metrics = None
        not_best_count = 0
        val_dataloader = self.meta_dataloaders[target_project]['test']
        
        maml = l2l.algorithms.MAML(model, lr=0.1, allow_nograd=True)
        opt = torch.optim.Adam(maml.parameters(), lr=self.config.learning_rate)
        for epoch in range(train_steps // valid_every):
            pbar = tqdm(range(valid_every))
            losses = []
            for iteration in pbar:
                opt.zero_grad()
                for project in source_projects:
                    sup_batch, qry_batch = next(iter(self.meta_dataloaders[project]['support'])), next(iter(self.meta_dataloaders[project]['query']))
                    task_model = maml.clone() # torch.clone() for nn.Modules
                    
                    outputs, labels = self.run_batch(task_model, sup_batch)
                    adaptation_loss = outputs.loss
                    task_model.adapt(adaptation_loss)  # computes gradient, update task_model in-place

                    outputs, labels = self.run_batch(task_model, qry_batch)
                    outputs.loss.backward()  # gradients w.r.t. maml.parameters()
                    losses.append(outputs.loss.item())
                opt.step()
                    
                pbar.set_description('Epoch = %d [loss=%.4f, min=%.4f, max=%.4f] %d' % (epoch, np.mean(losses), np.min(losses), np.max(losses), not_best_count))
                
            if epoch >= eval_start:

                bleu4, metrics = self.evaluate(self.model, val_dataloader)
                
                if best_bleu4 < bleu4:
                    best_bleu4 = bleu4
                    best_metrics = metrics
                    self.model.save_pretrained(save_dir)
                    not_best_count = 0
                else:
                    not_best_count += 1
                if not_best_count >= early_stop:
                    break
                print('[INFO]', best_bleu4, not_best_count)
        return best_metrics
            
    def train(self, model, target_project, sample_size, save_dir, max_epoch=5, early_stop=2, eval_start=1):
        self.model = model
        train_dataset = list()
        origin_dataset = self.meta_datasets[target_project]['support']
        for i in range(sample_size):
            train_dataset.append(origin_dataset[i])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = self.meta_dataloaders[target_project]['test']
        
        # Base Performancce
        self.optim = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        best_f1, best_metrics = self.evaluate(model, val_dataloader)
        self.model.save_pretrained(save_dir)
        not_best_count = 0
        for epoch in range(max_epoch):
            model.train()
            worker_count = 0
            self.optim.zero_grad()
            
            pbar = tqdm(train_dataloader)
            losses = []
            for batch in pbar:
                outputs, labels = self.run_batch(model, batch)
                loss = outputs.loss
                losses.append(loss.item())
                loss.backward()
                worker_count += 1
                if worker_count == self.num_worker:
                    self.optim.step()
                    self.optim.zero_grad()
                    worker_count = 0
                pbar.set_description('Epoch = %d [loss=%.4f, min=%.4f, max=%.4f]' % (epoch, np.mean(losses), np.min(losses), np.max(losses)))
                
            if worker_count > 0: 
                self.optim.step()
            if epoch >= eval_start:
                f1, metrics = self.evaluate(model, val_dataloader)
                if best_f1 < f1:
                    best_f1 = f1
                    best_metrics = metrics
                    not_best_count = 0
                    self.model.save_pretrained(save_dir)
                else:
                    not_best_count += 1
                if not_best_count >= early_stop:
                    break
                print('[INFO]', best_f1, not_best_count)
                    
        return best_metrics


def prefix_setup(seq2seq_model, config):
    for param in seq2seq_model.parameters():
        param.requires_grad = False

    for param in seq2seq_model.lm_head.parameters():
        param.requires_grad = True

    if seq2seq_model.model.encoder.layers[0].self_attn.use_prefix:
        print('prefix already inited')
        for layer_idx, layer in enumerate(seq2seq_model.model.encoder.layers):
            for param in layer.self_attn.prefix_network.parameters():
                param.requires_grad = True

        for layer_idx, layer in enumerate(seq2seq_model.model.decoder.layers):
            for param in layer.self_attn.prefix_network.parameters():
                param.requires_grad = True
            for param in layer.encoder_attn.prefix_network.parameters():
                param.requires_grad = True
    else:
        for layer_idx, layer in enumerate(seq2seq_model.model.encoder.layers):
            layer.self_attn.prefix_enable(config, config.prefix_seqlen, config.d_model, config.prefix_mid_dim, None)

        for layer_idx, layer in enumerate(seq2seq_model.model.decoder.layers):
            layer.self_attn.prefix_enable(config, config.prefix_seqlen, config.d_model, config.prefix_mid_dim, None)
            layer.encoder_attn.prefix_enable(config, config.prefix_seqlen, config.d_model, config.prefix_mid_dim, None)
        
    return seq2seq_model.to(device)


def save_metrics(all_metrics, config):
    projects = ['spring-boot', 'spring-framework', 'spring-security', 'guava', 'ExoPlayer', 'dagger', 'kafka', 'dubbo', 'flink']
    sorted_metrics = sorted(all_metrics.items(), key=lambda x: projects.index(x[0].split('|||')[1]) * 10000 + int(x[0].split('|||')[2]))
    json.dump(sorted_metrics, open('all_metrics_%s.json' % (config.model_name, ), 'w'), indent=2)

meta_datasets, code2id, comment2id = pickle.load(open('/tf/code_summarization/datasets/mtl/meta_datasets/meta_data_ijcai2022.pkl', 'rb'))

for dataset in meta_datasets.values():
    for split2 in ['support', 'query', 'test']:
        samples = dataset[split2]
        code_sequences = preprocess([sample['code'] for sample in dataset[split2]], code2id, config.code_max_len)
        comment_sequences = preprocess([sample['comment'] for sample in dataset[split2]], comment2id, config.comment_max_len)
        dataset[split2] = CustomDataset({'code': code_sequences, 'comment': comment_sequences})
        
id2comment = {v:k for k,v in comment2id.items()}


all_metrics = {}

all_projects = ['spring-boot', 'spring-framework', 'spring-security', 'guava', 'ExoPlayer', 'dagger', 'kafka', 'dubbo', 'flink']
project2sources = {
    'spring-boot': ['spring-framework', 'spring-security', 'guava'], 
    'spring-framework': ['spring-boot', 'spring-security', 'guava'], 
    'spring-security': ['spring-boot', 'spring-framework', 'guava'], 
    'guava': ['spring-framework', 'ExoPlayer', 'dagger'], 
    'ExoPlayer': ['guava', 'dagger', 'kafka'], 
    'dagger': ['guava', 'ExoPlayer', 'kafka'], 
    'kafka': ['dubbo', 'flink', 'guava'], 
    'dubbo': ['kafka', 'flink', 'guava'], 
    'flink': ['kafka', 'dubbo', 'guava'], 
}
trainer = MetaTrainer(config, id2comment, special_token_ids, device, meta_datasets)

all_projects = ['spring-boot', 'spring-framework', 'spring-security', 'guava', 'ExoPlayer', 'dagger', 'kafka', 'dubbo', 'flink']
for target_project in all_projects:
    source_projects = project2sources[target_project]
    print('------------', target_project, '----------------')
    print(source_projects)
    
    test_dataloader = trainer.meta_dataloaders[target_project]['test']
    model = TransformerForConditionalGeneration.from_pretrained(load_model_from, config=config).to(device)
    model = prefix_setup(model, config).to(device)
    
    meta_save_dir = './cpts/meta/%s' % (config.model_name, )
    if not os.path.exists(meta_save_dir):
        os.makedirs(meta_save_dir)

    best_metrics = trainer.meta_train(model, target_project, source_projects, meta_save_dir, 
                                     train_steps=12000, inner_train_steps=1,
                                      valid_steps=200, inner_valid_steps=4, 
                                      valid_every=5, eval_start=0, early_stop=10)
    print((config.model_name, target_project, 0))
    for metric, score in sorted(best_metrics.items(), key = lambda x: x[0]):
        print(metric, score)
        
    all_metrics['%s|||%s|||%s' % (config.model_name, target_project, 0)] = best_metrics
    save_metrics(all_metrics, config)
    
    for sample_size in [10, 100]:
        print('==============', sample_size, '============')
        project_save_dir = './cpts/target/%s' % (config.model_name, )
        if not os.path.exists(project_save_dir):
            os.makedirs(project_save_dir)
        
        model = TransformerForConditionalGeneration.from_pretrained(meta_save_dir, config=config)
        model = prefix_setup(model, config).to(device)
            
        best_metrics = trainer.train(model, target_project, sample_size, project_save_dir, 
                                     max_epoch=200, early_stop=10, eval_start=0)
        print((config.model_name, target_project, sample_size))
        for metric, score in sorted(best_metrics.items(), key = lambda x: x[0]):
            print(metric, score)

        all_metrics['%s|||%s|||%s' % (config.model_name, target_project, sample_size)] = best_metrics
        save_metrics(all_metrics, config)
        

        model = TransformerForConditionalGeneration.from_pretrained(project_save_dir, config=config)
        model = prefix_setup(model, config).to(device)
        y_pred = []
        y_true = []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader):
                outputs, labels = trainer.run_batch(model, batch, is_eval=True)
                preds = trainer.ids_to_clean_text(outputs.cpu())
                targets = trainer.ids_to_clean_text(labels.cpu())

                y_pred.extend(preds)
                y_true.extend(targets)

        
        json.dump((y_pred, y_true), open('predicts/%s_MPCOS_%s.json' % (target_project, sample_size), 'w'))
        trainer.calc_generative_metrics(y_pred, y_true)

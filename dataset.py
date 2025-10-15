from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from typing import List
import pandas as pd
import ast


MAIN_REASONS = [
    'Too narrow or specific',
    'Too broad or vague',
    'Lacks necessary context',
    'Ambiguous or poorly defined',
    'Unclear topic or focus',
    'Requires specialized knowledge',
    'Not a general knowledge query',
    'No clear answer available',
    'Not a standard question',
    'Unconventional format or structure'
]

class Dataset(Dataset):
    def __init__(self, dataPath, tokenizer, dataset_name='train', expand=None):
        self.dataPath = dataPath
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self._pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self.tokenizer.pad_token_type_id,
            "special_tokens_mask": 1,
        }
        self.experiment = 'reasons'
        self.expand = expand
        self.read()
    
    def __getitem__(self, index):
        keys = [*self.queryDic]
        queryID = keys[index]
        queryText = self.queryDic[queryID]
        output = self.tokenizer.encode_plus(
                queryText,
                padding=True, truncation=True,
                return_tensors="pt",)
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        token_type_ids = None
        
        if self.expand == 'reasons':
            reasons = self.query_reasons[queryID]
            reasons = torch.tensor(reasons)
            input_ids = torch.cat((input_ids, reasons.unsqueeze(0)), dim=1)
            
            ones = torch.ones(reasons.shape[0], dtype=torch.long)
            attention_mask = torch.cat((attention_mask, ones.unsqueeze(0)), dim=1)
        
            zeros = torch.zeros(reasons.shape[0], dtype=torch.long)
            token_type_ids = torch.cat((token_type_ids, zeros.unsqueeze(0)), dim=1)
        
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'token_type_ids': token_type_ids
        }
        assert all(v.size(0) == 1 for v in output.values())


        return output , queryID, [self.MAPScore[queryID]]

    def __len__(self):
        return (len(self.queryDic))
    
    @staticmethod
    def ones_expanding(reasons):
        reasons = ast.literal_eval(reasons)
        reasons = [entity for flag, entity in zip(reasons, MAIN_REASONS) if flag == 1]
        reasons = ' '.join(reasons)
        return reasons
    
    @staticmethod
    def oneHot_expanding(reasons):
        reasons = ast.literal_eval(reasons)
        reasons = [str(flag) + ' ' + entity for flag, entity in zip(reasons, MAIN_REASONS)]
        reasons = '[SEP]'.join(reasons)
        return reasons
    
    @staticmethod
    def oneHot_1s_expanding(reasons):
        reasons = ast.literal_eval(reasons)
        reasons = [str(flag) + ' ' + entity for flag, entity in zip(reasons, MAIN_REASONS) if flag == 1]
        reasons = ' '.join(reasons)
        return reasons
    
    def get_reasons_ds(self):
        dataset = self.dataset_name
        dataset = dataset.lower()
        if dataset == 'train-onehot' or dataset == 'onehot-1s':

            path = './reasoning_by_llm/oneHot_reasons.train.v1.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'train':
            path = './reasoning_by_llm/oneHot_reasons.train.v1.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl1920':
            path = './reasoning_by_llm/oneHot_reasons.1920.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl19':
            path = './reasoning_by_llm/oneHot_reasons.19.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl20':
            path = './reasoning_by_llm/oneHot_reasons.20.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl21':
            path = './reasoning_by_llm/oneHot_reasons.21.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl22':
            path = './reasoning_by_llm/oneHot_reasons.22.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dl2122':
            path = './reasoning_by_llm/oneHot_reasons.2122.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dev':
            path = './reasoning_by_llm/oneHot_reasons.dev.small.qwen3:8b.oneHot-reasoning.txt'
        elif dataset == 'dlhard':
            path = './reasoning_by_llm/oneHot_reasons.hard.qwen3:8b.oneHot-reasoning.txt'
        
        if '.txt' in path:
            queries_df = pd.read_csv(path, sep='\t', names=['ds', 'qid', 'query', 'reasons'])
        else:
            queries_df = pd.read_csv(path)
        queries_df = queries_df[queries_df['qid'].isin(self.MAPScore.keys())]
        print(f'Expand -> {self.expand}')
        

        if self.expand == 'onehot':
            reasons = queries_df['reasons'].apply(ast.literal_eval)
            reasons = reasons.apply(lambda x: ' '.join(map(str, x)))
            queries_df['input'] = queries_df['query'] + ' ' + reasons
        else:
            raise Exection('Expand not supported')

        
        output = dict(zip(list(queries_df['qid'].values), list(queries_df['input'].values)))
        print('output\n', queries_df['input'].values[:3])
        return output

    

    def read(self):
        import csv
        self.queryDic = {}
        self.collectionDic = {}
        self.MAPScore = {}
      
        
        
        if 'train' in self.dataset_name:
            MAPScore_file = open(self.dataPath + "/50/train_query_mrr.tsv")
        elif self.dataset_name == 'dl1920':
            MAPScore_file = open(self.dataPath + "/dl1920_ndcg10.tsv")
        elif self.dataset_name == 'dl19':
            MAPScore_file = open(self.dataPath + "/dev2019_query_NDCG.tsv")
        elif self.dataset_name == 'dl20':
            MAPScore_file = open(self.dataPath + "/dev2020_query_NDCG.tsv")
        elif self.dataset_name == 'dl21':
            MAPScore_file = open(self.dataPath + "/dl2021_ndcg10")
        elif self.dataset_name == 'dl22':
            MAPScore_file = open(self.dataPath + "/dl2022_ndcg10")
        elif self.dataset_name == 'dl2122':
            MAPScore_file = open(self.dataPath + "/dl202122_ndcg10.tsv")
        elif self.dataset_name == 'dlhard':
            MAPScore_file = open(self.dataPath + "/dlhard_ndcg10.tsv")
            
        elif self.dataset_name == 'dev':
            MAPScore_file = open(self.dataPath + "/dev_query_mrr.tsv")
        
        read_tsv = csv.reader(MAPScore_file, delimiter="\t")
        for row in read_tsv:
            self.MAPScore[int(row[0])] = float(row[1])
            
        self.queryDic = self.get_reasons_ds()


        print('len self.queryDic', len(self.queryDic))
        print('len self.MAPScore', len(self.MAPScore))

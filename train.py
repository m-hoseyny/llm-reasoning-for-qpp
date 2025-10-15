import json
import sys, os
from transformers import BertTokenizer
from torch.utils.data import DataLoader, random_split
from PadCollate import PadCollate
from dataset import Dataset
from losses import QPPLoss
import torch
from model import DRAQModel
from transformers import get_linear_schedule_with_warmup
import time
from utils import format_time, computeMetric
import random
import numpy as np
import warnings
import tqdm
import wandb
warnings.filterwarnings("ignore")

if torch.cuda.is_available():device = torch.device("cuda")
else:device = torch.device("cpu")

def setup_wandb(config):
    # Disable wandb's Sentry error reporting to run on clusters
    os.environ["WANDB_ERROR_REPORTING"] = "False"
    sample_size = 0

    model_name = config['bertModel'].split('/')[-1] + '_' + config['expand']
    kwargs = {'name': f'{model_name}', 'project': f'QPP-GTE-QWEN',
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'config': config,
              'mode': 'online'
              }
    wandb.init(**kwargs)
    wandb.save('*.txt')

def train(config):
    tokenizer = BertTokenizer.from_pretrained(config['bertModel'])
    model = DRAQModel(config,device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=config['epsilon_parameter'])

    random.seed(config['seed_val'])
    np.random.seed(config['seed_val'])
    torch.manual_seed(config['seed_val'])
    torch.cuda.manual_seed_all(config['seed_val'])

    expand = 'onehot'
    config['expand'] = expand
    print(f'------\n{expand} -> Training\n------')
    datasetTrain = Dataset(dataPath=config['dataPath'], tokenizer=tokenizer, dataset_name='train-onehot', expand=expand)

    print('{:>5,} training samples'.format(len(datasetTrain)))

    trainDataloader = DataLoader(dataset=datasetTrain, batch_size=config['batch'], shuffle=True,
                                 collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id))


    total_steps = len(trainDataloader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['num_warmup_steps'], num_training_steps=total_steps)
    setup_wandb(config)

    for epoch_i in range(0, config['epochs']):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['epochs']))
        print('Training...')

        t0 = time.time()


        model.train()
        for step, batch in tqdm.tqdm(enumerate(trainDataloader)):
            inputs, query, MAPScore = batch
            bsz, gsz, _ = inputs["input_ids"].size()

            inputs = {
                k: v.view(bsz * gsz, -1)
                for k, v in inputs.items()
                }
            inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "token_type_ids": inputs.get("token_type_ids")
            }
            model.zero_grad()
            qpp_logits = model.forward(inputs['input_ids'], inputs['attention_mask'],inputs['token_type_ids'])

            MAPScore = np.asarray(MAPScore)
            MAPScore = torch.from_numpy(MAPScore.astype('float32'))
            qpp_loss = QPPLoss(device).loss(qpp_logits, MAPScore)
            qpp_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            wandb.log({
                'train/qpp_loss': qpp_loss.item(),
                'train/step': step * (epoch_i + 1)
            })
        model_name = config['bertModel'].split('/')[-1]
        file_name = 'output/DRAQ_{}_{}_{}_{}_{}.model'.format(model_name, expand, config['batch'], config['learning_rate'], epoch_i)
        torch.save(model, config['outputPath'] + file_name)

if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
        jsonfile.close()
    train(config)
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from PadCollate import PadCollate
from dataset import Dataset
import torch
from utils import computeMetric
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():device = torch.device("cuda")
else:device = torch.device("cpu")


def test(config, dataset_name='dl19'):
    tokenizer = BertTokenizer.from_pretrained(config['bertModel'])
    expand = 'onehot'
    model_name = 'DRAQ-base-uncased_onehot_16_0.0001_1.model'
    model = torch.load(f'model-query-reasons/{model_name}', weights_only=False)
    model.to(device)  

    random.seed(config['seed_val'])
    np.random.seed(config['seed_val'])
    torch.manual_seed(config['seed_val'])
    torch.cuda.manual_seed_all(config['seed_val'])
    

    datasetVal = Dataset(dataPath=config['dataPath'], tokenizer=tokenizer, dataset_name=dataset_name, expand=expand)

    print('{:>5,} validation samples'.format(len(datasetVal)))

    valDataloader = DataLoader(dataset=datasetVal, batch_size=config['batch'], shuffle=True,
                               collate_fn=PadCollate(tokenizer.pad_token_id, tokenizer.pad_token_type_id))

    QPP = {}
    MAP = {}
    output = open('DRAQOutPut.tsv','w')
    for batch in valDataloader:
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
        with torch.no_grad():
            qpp_logits = model.forward(inputs['input_ids'], inputs['attention_mask'],inputs['token_type_ids'])

        for q in range(0, len(query)):
            MAP[query[q]] = float(MAPScore[q][0])
            QPP[query[q]] = float(qpp_logits[q].item())
            output.write(str(query[q])+'\t'+str(qpp_logits[q].item())+'\n')


    pearsonr, pearsonp, kendalltauCorrelation, kendalltauPvalue, spearmanrCorrelation, spearmanrPvalue = computeMetric(QPP,MAP)
    print(f'------\n{dataset_name} -> Testing\n------')
    print("  pearsonrCorrelation: {0:.3f}".format(pearsonr))
    print("  pearsonrPvalue: {0:.3f}".format(pearsonp))
    print("  kendalltauCorrelation: {0:.3f}".format(kendalltauCorrelation))
    print("  kendalltauPvalue: {0:.3f}".format(kendalltauPvalue))
    print("  spearmanrCorrelation: {0:.3f}".format(spearmanrCorrelation))
    print("  spearmanrPvalue: {0:.3f}".format(spearmanrPvalue))


if __name__ == '__main__':
    with open("config.json", "r") as jsonfile:
        config = json.load(jsonfile)  # Reading the file
        print(config)
        jsonfile.close()
    for dataset_name in ['dev', 'dlhard', 'dl19', 'dl20', 'dl21', 'dl22']:
        test(config, dataset_name)
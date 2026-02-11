import pandas as pd
import os, sys
import argparse
import copy
import ast
import openai
import time
import tqdm
import json
import re

Hard2PerformeJustQuery = [
                {
                    "role": "system",
                    "content": (
                        "You are AssessmentLLM, an intelligent assistant that can label a list of atomic reasons based on"
                        "if they are captured by a given query."
                        "\n/no_think"
                    )
                },
                {
                    "role": "user",
                    "content": ( 'Based on the query and passage, label each of the 10 reasons either as '
                                'captured or not captured using the following criteria. '
                                'A reason that is captured in the query should be labeled as captured (1). '
                                'A reason that is not captured at all, label it as not captured (0). '
                                'Return the list of 10 labels in a Pythonic list format (type: List[int]). '
                                'The list should be in the same order as the input reasons. '
                                'Ensure a label for each reasons. \n' 
                                'Query: {}\n'
                                'Reasons List: {} \n'
                                'Only return the list of labels (List[int]). Do not explain.\n'
                                'The len of the list must be 10! Do not return more or less than 10 labels.\n'
                                'Just return list of 0/1. Not the reasons list.\n'
                                'Do not explain anything. Just return a list of int.\n'
                                'Labels:') 
                }
                ]

reasons_list = [
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

UNIFIED_REASONS = json.dumps(reasons_list)
print('UNIFIED_REASONS: ', UNIFIED_REASONS)
def load_df(path_):
    df = pd.read_csv(path_, 
                         sep='\t', 
                         names=['qid', 'query'],
                         engine='pyarrow')
    return df


def extract_int_lists(text):
    # Match content inside square brackets
    matches = re.findall(r"\[(.*?)\]", text)
    result = []
    for match in matches:
        # Find all integers inside each bracketed group
        numbers = re.findall(r"-?\d+", match)
        result.append([int(num) for num in numbers])
    return result


def extract_reasons(df, dataset_name, model_name, task):
    
    client = openai.OpenAI(api_key='ollama', 
                           base_url='http://localhost:11434/v1')
    
    seen_set = set()
    previous_list = []
    results = []
    messages = Hard2PerformeJustQuery
    save_path = f'reasoning_by_llm/oneHot_reasons.{dataset_name}.{model_name}.{task}.with_thinking.txt'
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            seen_set = set(line.split('\t')[2] for line in f)
    
    for i, ds_row in df.iterrows():
        Done = False
        query = ds_row['query']
        timeout = 0
        if query in seen_set:
            continue

        while not Done and timeout < 5:
            start_time = time.time()
            seen_set.add(query)
            q = ds_row['qid']
            response = None
            local_message = copy.deepcopy(messages)
            local_message[1]['content'] = local_message[1]['content'].format(query, UNIFIED_REASONS)
            print('Processing query: ', query)
            try:
                response = client.chat.completions.create(
                    model= model_name,
                    messages=local_message,
                    temperature=0,
                    top_p=1,
                    extra_body={"think": False}
                )
                data = response.choices[0].message.content.split('</think>')[-1].strip()
                thinking = response.choices[0].message.content.split('</think>')[0].strip()
                data = data.replace('`', '').strip()
                one_hot_encoding = extract_int_lists(data)[0]
                print('Data is :', data)
                # one_hot_encoding = ast.literal_eval(data)
                if len(one_hot_encoding) != 10:
                    print(f'Warning: len oneHot = {len(one_hot_encoding)}, Data: {one_hot_encoding}')
                    timeout += 1
                elif not all(isinstance(x, int) for x in one_hot_encoding):
                    print(f'Warning: oneHot is not int')
                    timeout += 1
                else:
                    Done = True
            except Exception as e:
                print(response)
                print(f"Error processing QID {q}: {e}")
                timeout += 1
                continue
        
        results.append({'query': query, 'qid': q, 'one_hot_encoding': one_hot_encoding, 'thinking': thinking})
        with open(save_path,'a') as output:
            output.write(f'{dataset_name}\t{q}\t{query}\t{one_hot_encoding}\t{thinking}\n')
        pd.DataFrame(results).to_csv(save_path.replace('.txt', '.csv'), index=False)
        spent_time = time.time() - start_time
        eta = spent_time * (len(df) - i) / 60
        print('[{}/{}] (Spent: {:.2f} sec) (ETA: {:.2f} min) : {} {} {} {} '.format(i, len(df), spent_time, eta,
                                                    q, query, one_hot_encoding ,len(one_hot_encoding)))
        
        
if __name__ == '__main__':
    model_name = 'mistral:latest'
    task = 'oneHot-reasoning'
    for dataset_name in ['train', 'hard', '2020', '2019', '2021', '2022' 'dev.small']:
        if dataset_name == 'train':
            df_path = 'dataset/queries.train_filtered.tsv'
        else:
            df_path = f'dataset/test{dataset_name}-queries-filterd.tsv'
        df = load_df(df_path)
        extract_reasons(df, dataset_name, model_name, task)
    
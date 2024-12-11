import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd


import ssl
import urllib.request

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"



def load_TAG_dataset(file_path, num_samples=None):

    if not os.path.exists(file_path):
        raise ValueError(f"Test file {file_path} does not exist.")

    return load_csv(file_path, num_samples)


def load_csv(file_path, num_samples):
    """
    Data format for Children dataset:
    prompt,raw_text,correct_category,false_categories,node_id,neighbour
    Example row:
    "Your task is to classify the given text into one of the following categories: ...",
    "Description: Collection of Poetry; Title: The golden treasury of poetry",
    "Literature & Fiction",
    "['Animals', 'Growing Up & Facts of Life', ...]",
    0,
    "[5472, 14293, 15164, 26542, 33933]"
    """
    import pandas as pd
    import ast

    # Read the CSV file
    df = pd.read_csv(file_path)

    # If num_samples is specified, limit the number of rows
    if num_samples is not None:
        df = df.head(num_samples)

    # Convert the data into the desired format
    list_data_dict = []
    for idx in range(len(df)):
        item = {
            "prompt": df["prompt"][idx],
            "raw_text": df["raw_text"][idx],
            "correct_category": df["correct_category"][idx],
            "false_categories": ast.literal_eval(df["false_categories"][idx]),
            "node_id": df["node_id"][idx],
            "neighbour": ast.literal_eval(df["neighbour"][idx])
        }
        list_data_dict.append(item)

    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        #print(f'File {file} exists, use existing file.')
        return path

    #print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
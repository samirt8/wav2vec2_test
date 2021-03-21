import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

from datasets import load_metric
import soundfile as sf

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Trainer, TrainingArguments

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

def compute_metrics(label, pred):
    """
    compute WER here
    """
    return None

with open("...", "r") as f:
    reference = f.read()

model = Wav2Vec2ForCTC.from_pretrained("/media/nas/samir-data/wav2vec2_models/checkpoint-47000-benchmark")
processor = Wav2Vec2Processor.from_pretrained("/media/nas/samir-data/wav2vec2_models/checkpoint-47000-benchmark")

# initialize the prediction
prediction = ""
with SoundFile(file, 'r', 16000) as f:
    for block in f.blocks(blocksize=50000, overlap=100):
        input_dict = processor(block, return_tensors="pt", padding=True)
        logits = model(input_dict.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        prediction += processor.decode(pred_ids)

print("Prediction:")
print(prediction)

print("\nReference:")
print(reference[0])

#print("\nWer Metric:")

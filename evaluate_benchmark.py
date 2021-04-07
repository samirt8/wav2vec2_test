import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

from datasets import load_metric
import soundfile as sf

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
    Wav2Vec2Tokenizer
from transformers import Trainer, TrainingArguments

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

from data_benchmark import AudioDataset


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

test_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr_v6.1/cv-corpus-6.1-2020-12-11/fr/clips"
test_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr_v6.1/cv-corpus-6.1-2020-12-11/fr/test.tsv"
test_set = AudioDataset(test_annotation_file, test_data_folder)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=1)

processor = Wav2Vec2Processor.from_pretrained("/media/nas/samir-data/wav2vec2_models/wav2vec2_trained_on_commonvoice_v2_fr")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("/media/nas/samir-data/wav2vec2_models/wav2vec2_trained_on_commonvoice_v2_fr")

model = Wav2Vec2ForCTC.from_pretrained("/media/nas/samir-data/wav2vec2_models/wav2vec2_trained_on_commonvoice_v2_fr")
checkpoint = torch.load("/media/nas/samir-data/wav2vec2_models/wav2vec2_trained_on_commonvoice_v2_fr/pytorch_model.bin")

from collections import OrderedDict
new_checkpoint = OrderedDict()
for k, v in checkpoint.items():
    new_k = k[7:]
    new_checkpoint[new_k] = v

model.load_state_dict(checkpoint)
model.eval()

wer_metric = load_metric("wer")

# initialize the prediction
wers = []
for audio_file in test_generator:
    input_dict = processor(np.squeeze(audio_file["input_values"], 0), return_attention_mask=True, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        pred = model(input_dict.input_values, attention_mask=input_dict.attention_mask)
        logits = pred.logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        prediction = tokenizer.decode(pred_ids)

        reference = "".join([" " if x == "|" else x for x in tokenizer.convert_ids_to_tokens(audio_file["labels"][0], skip_special_tokens=True)])

        print("Prediction:")
        print(prediction)

        print("Reference:")
        print(reference)

        wers.append(wer_metric.compute(predictions=[prediction], references=[reference]))
        print("\n")
        

print("\nWer Metric: ", sum(wers)/len(wers))

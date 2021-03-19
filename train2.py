import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

from datasets import load_metric

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer
from transformers import Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.autograd.detect_anomaly()

from model import ASR_CTC
from data2 import AudioDataset


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


if __name__ == '__main__':

    epochs = 1
    batch_size = 1
    MAX_LEN = 32
    learning_rate = 5e-4
    hyperparameters = {
        'batch_size': batch_size,
        'shuffle': True
    }
    train_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    train_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/train_subset.tsv"
    validation_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    validation_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/dev_subset.tsv"
    save_path = "/media/nas/samir-data/wav2vec2_models"
    #os.mkdir(save_path)
    with open(save_path+'/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    wer_metric = load_metric("wer")

    #Generators
    training_set = AudioDataset(train_annotation_file, train_data_folder, MAX_LEN)
    training_generator = torch.utils.data.DataLoader(training_set, **hyperparameters)

    validation_set = AudioDataset(validation_annotation_file, validation_data_folder, MAX_LEN)
    validation_generator = torch.utils.data.DataLoader(validation_set, **hyperparameters)

    print('LOAD TOKENIZER...')
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                      do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    nb_labels = 37

    print('INITIALIZING MODEL...')
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
    #data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    #wer_metric = load_metric("wer", script_version="master")

    training_args = TrainingArguments(
        output_dir="/media/nas/samir-data/wav2vec2_models/wav2vec2-large-xlsr-french-demo",
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    print('TRAINING ALL LAYER...')
    wav2vec2_model.freeze_feature_extractor()
    trainer = Trainer(
        model=wav2vec2_model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=training_set,
        eval_dataset=validation_set,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

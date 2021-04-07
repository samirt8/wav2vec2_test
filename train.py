import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.autograd.detect_anomaly()

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_metric

from data import AudioDataset


def compute_metrics(y_pred, y_true):
    wer = wer_metric.compute(predictions=y_pred, references=y_true)
    return {"wer": wer}


def collate_fn(features):
    # split inputs and labels since they have to be of different lenghts and need
    # different padding methods
    input_features = [{"input_values": feature["input_values"]} for feature in features]
    label_features = [{"input_ids": feature["labels"]} for feature in features]

    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
        )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt"
        )

    # replace padding with -100 to ignore loss correctly
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    batch["labels"] = labels

    return batch


def validate(model, epoch, validation_generator, save_path, train_loss,
             best_val_loss, best_model_path):
    val_losses = []
    wers = []

    for batch in validation_generator:
        with torch.no_grad():
            inputs, attentions, labels = batch["input_values"].to(device="cuda"), batch["attention_mask"].to(device="cuda"), batch["labels"].to(device="cuda")
            output = model(inputs, attention_mask=attentions, labels=labels)

            y_pred = torch.argmax(output.logits, dim=-1).cpu().data.numpy()
            y_true = labels.cpu().data.numpy()

            print("y_pred : ", tokenizer.batch_decode(y_pred))
            print("y_true : ", tokenizer.batch_decode(y_true))

            val_loss = output.loss
            val_losses.append(val_loss.cpu().data.numpy())
            wers.append(compute_metrics(tokenizer.batch_decode(y_pred), tokenizer.batch_decode(y_true)))

    val_loss = np.mean(val_losses)
    wer = np.mean([list(d.values()) for d in wers])

    model_path = save_path+'/model_wer'+str(wer)
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_wer = wer
        best_model_path = model_path

    return best_val_loss, best_wer, best_model_path


def train(model, optimizer, epochs, training_generator, validation_generator, save_path, best_val_loss=1e9):

    print_every = len(training_generator)
    best_model_path = None
    model.train(mode=True)
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1

        for batch in training_generator:
            inputs, attentions, labels = batch["input_values"].to(device="cuda"), batch["attention_mask"].to(device="cuda"), batch["labels"].to(device="cuda")

            inputs.requires_grad = False
            attentions.requires_grad = False
            labels.requires_grad = False
            output = model(inputs, attention_mask=attentions, labels=labels)

            labels[labels == -100] = processor.tokenizer.pad_token_id

            y_pred = torch.argmax(output.logits, dim=-1).cpu().data.numpy()
            y_true = labels.cpu().data.numpy()

            print("y_pred : ", tokenizer.batch_decode(y_pred))
            print("y_true : ", tokenizer.batch_decode(y_true))

            loss = output.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path, best_wer = validate(model, e, validation_generator,
                    save_path, train_loss, best_val_loss, best_model_path)
                model.train()
                pbar = tqdm(total=print_every)
            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path, best_wer = validate(model, e, validation_generator,
        save_path, train_loss, best_val_loss, best_model_path)
        model.train()
        if e < epochs-1:
            pbar = tqdm(total=print_every)

    #model.load_state_dict(torch.load(best_model_path))
    #model.eval()

    return model, optimizer, best_val_loss, best_wer

if __name__ == '__main__':

    epochs = 1
    batch_size = 4
    learning_rate = 4e-5
    hyperparameters = {
        'batch_size': batch_size,
        'shuffle': True,
        'collate_fn': collate_fn
    }
    train_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    train_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/train.tsv"
    validation_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    validation_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/test1.tsv"
    save_path = "/media/nas/samir-data/wav2vec2_models"

    print('LOAD TOKENIZER...')
    tokenizer = Wav2Vec2CTCTokenizer("vocab_v2.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                      do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    nb_labels = len(tokenizer.get_vocab())

    #WER Metric
    wer_metric = load_metric("wer")
    #Initialize best_wer
    best_wer=1.0

    #Generators
    training_set = AudioDataset(train_annotation_file, train_data_folder)
    training_generator = torch.utils.data.DataLoader(training_set, **hyperparameters)

    validation_set = AudioDataset(validation_annotation_file, validation_data_folder)
    validation_generator = torch.utils.data.DataLoader(validation_set, **hyperparameters)

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
).cuda()

    print('TRAINING ALL LAYER...')
    optimizer = optim.AdamW(wav2vec2_model.parameters(), lr=learning_rate)
    wav2vec2_model.freeze_feature_extractor()
    wav2vec2_model, optimizer, best_val_loss, best_wer = train(wav2vec2_model, optimizer, epochs,
        training_generator, validation_generator, save_path)

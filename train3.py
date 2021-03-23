import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import os
import json
from datetime import datetime

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_metric

import torch
from torch import nn, optim
import torch.nn.functional as F
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import Trainer, TrainingArguments

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
torch.autograd.detect_anomaly()

from model import ASR_CTC
from data2 import AudioDataset


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        if input_features is None:
            return None

        else:
            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def validate(model, criterion, epoch, epochs, validation_generator, save_path, train_loss,
             best_val_loss, best_model_path):
    val_losses = []

    for inputs in tqdm(validation_generator, total=len(validation_generator)):
        with torch.no_grad():
            inputs, attentions, labels = inputs["input_values"].cuda(), inputs["attention_mask"].cuda(), inputs["labels"].cuda()
            output = model(inputs, attentions).logits
            output1 = output.view(-1, batch_size, nb_labels)

            y_pred = torch.argmax(output, dim=-1).cpu().data.numpy()
            y_true = labels.cpu().data.numpy()

            output_lengths = torch.full(size=(batch_size,), fill_value=output1.shape[0], dtype=torch.long)
            labels_lengths = torch.randint(low=y_true.shape[2]-5, high=y_true.shape[2], size=(batch_size,), dtype=torch.long)
            val_loss = criterion(output1, labels[0], output_lengths, labels_lengths)
            val_losses.append(val_loss.cpu().data.numpy())

            print("y_true : ", tokenizer.batch_decode(y_true[0]))
            print("y_pred : ", tokenizer.decode(y_pred[0]))

    val_loss = np.mean(val_losses)

    improved = ''

    model_path = save_path+'/model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path

    progress_path = save_path+'/progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('time;epoch;training loss;loss;accuracy;''\n')

    with open(progress_path, 'a') as f:
        f.write('{};{};{:.4f};{:.4f}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch+1,
            "Loss: {:.4f}".format(train_loss),
            "Val Loss: {:.4f}".format(val_loss)
            ))

    return best_val_loss, best_model_path


def train(model, optimizer, criterion, epochs, training_generator, validation_generator, save_path, best_val_loss=1e9):

    print_every = len(training_generator)
    best_model_path = None
    model.train(mode=True)
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1

        for inputs in training_generator:

            inputs, attentions, labels = inputs["input_values"], inputs["attention_mask"], inputs["labels"]

            inputs.requires_grad = False
            labels.requires_grad = False
            output = model(inputs, attentions).logits

            y_pred = torch.argmax(output, dim=-1).cpu().data.numpy()
            y_true = labels.cpu().data.numpy()

            print("y_true : ", tokenizer.batch_decode(y_true[0]))
            print("y_pred : ", tokenizer.decode(y_pred[0]))

            output1 = output.view(-1, batch_size, nb_labels)
            output_lengths = torch.full(size=(batch_size,), fill_value=output1.shape[0], dtype=torch.long)
            labels_lengths = torch.randint(low=y_true.shape[2]-5, high=y_true.shape[2], size=(batch_size,), dtype=torch.long)
            loss = criterion(output1, labels[0], output_lengths, labels_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, epochs, validation_generator,
                    save_path, train_loss, best_val_loss, best_model_path)
                model.train()
                pbar = tqdm(total=print_every)
            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, criterion, e, epochs, validation_generator,
            save_path, train_loss, best_val_loss, best_model_path)
        model.train()
        if e < epochs-1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss

if __name__ == '__main__':

    epochs = 1
    batch_size = 1
    MAX_LEN = 32
    learning_rate = 5e-5
    hyperparameters = {
        'batch_size': batch_size,
        'shuffle': True
    }
    train_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr_v6.1/cv-corpus-6.1-2020-12-11/fr/clips"
    train_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr_v6.1/cv-corpus-6.1-2020-12-11/fr/train.tsv"
    validation_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    validation_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/dev1.tsv"
    save_path = "/media/nas/samir-data/wav2vec2_models"
    #os.mkdir(save_path)
    with open(save_path+'/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    #Generators
    training_set = AudioDataset(train_annotation_file, train_data_folder, MAX_LEN)
    training_generator = torch.utils.data.DataLoader(training_set, **hyperparameters)

    validation_set = AudioDataset(validation_annotation_file, validation_data_folder, MAX_LEN)
    validation_generator = torch.utils.data.DataLoader(validation_set, **hyperparameters)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    #print('LOAD TOKENIZER...')
    tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                      do_normalize=True, return_attention_mask=True)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    nb_labels = 36

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

    #wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("/media/nas/samir-data/wav2vec2_models/checkpoint-18000")
    #processor = Wav2Vec2Processor.from_pretrained("/media/nas/samir-data/wav2vec2_models/checkpoint-18000")

    wav2vec2_model.freeze_feature_extractor()
    wav2vec2_model_cuda = nn.DataParallel(wav2vec2_model.cuda())

    optimizer = optim.AdamW(wav2vec2_model_cuda.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss(reduction="mean", blank=0)

    wer_metric = load_metric("wer")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print('TRAINING ALL LAYER...')
    #wav2vec2_model_cuda, optimizer, best_val_loss = train(wav2vec2_model_cuda, optimizer, criterion, epochs, training_generator, validation_generator, save_path)

    training_args = TrainingArguments(
    output_dir=save_path,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)

    trainer = Trainer(
    model=wav2vec2_model_cuda,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=training_set,
    eval_dataset=validation_set,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator
)
trainer.train()

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
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

from sklearn import metrics  # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from model import ASR_CTC
from data import AudioDataset


def validate(model, criterion, epoch, epochs, iteration, iterations, data_loader_valid, save_path, train_loss,
             best_val_loss, best_model_path):
    val_losses = []
    val_accs = []
    val_f1s = []

    for inputs, labels in tqdm(data_loader_valid, total=len(data_loader_valid)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            val_loss = criterion(output, labels)
            val_losses.append(val_loss.cpu().data.numpy())

            #y_pred = output.argmax(dim=1).cpu().data.numpy().flatten()
            y_pred = torch.argmax(model(inputs).logits, dim=-1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()
            val_accs.append(metrics.accuracy_score(y_true, y_pred))
            val_f1s.append(metrics.f1_score(y_true, y_pred, average=None, labels=label_vals))

    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    val_f1 = np.array(val_f1s).mean(axis=0)

    improved = ''

    # model_path = '{}model_{:02d}{:02d}'.format(save_path, epoch, iteration)
    model_path = save_path+'model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path

    f1_cols = ';'.join(['f1_'+key for key in label_keys])

    progress_path = save_path+'progress.csv'
    if not os.path.isfile(progress_path):
        with open(progress_path, 'w') as f:
            f.write('time;epoch;iteration;training loss;loss;accuracy;'+f1_cols+'\n')

    f1_vals = ';'.join(['{:.4f}'.format(val) for val in val_f1])

    with open(progress_path, 'a') as f:
        f.write('{};{};{};{:.4f};{:.4f};{:.4f};{}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch+1,
            iteration,
            train_loss,
            val_loss,
            val_acc,
            f1_vals
            ))

    print("Epoch: {}/{}".format(epoch+1, epochs),
          "Iteration: {}/{}".format(iteration, iterations),
          "Loss: {:.4f}".format(train_loss),
          "Val Loss: {:.4f}".format(val_loss),
          "Acc: {:.4f}".format(val_acc),
          "F1: {}".format(f1_vals),
          improved)

    return best_val_loss, best_model_path

def train(model, optimizer, criterion, epochs, training_generator, validation_generator, save_path, best_val_loss=1e9):

    print_every = len(data_loader_train)//((iterations+1))
    clip = 5
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1
        iteration = 1

        for inputs, labels in training_generator:

            inputs, labels = inputs.cuda(), labels.cuda()
            inputs.requires_grad = False
            labels.requires_grad = False
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations, validation_generator,
                    save_path, train_loss, best_val_loss, best_model_path)
                model.train()
                pbar = tqdm(total=print_every)
                iteration += 1
            counter += 1

        pbar.close()
        model.eval()
        best_val_loss, best_model_path = validate(model, criterion, e, epochs, iteration, iterations, validation_generator,
            save_path, train_loss, best_val_loss, best_model_path)
        model.train()
        if e < epochs-1:
            pbar = tqdm(total=print_every)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    return model, optimizer, best_val_loss

if __name__ == '__main__':

    epochs = 1
    batch_size = 128
    learning_rate_top = 3e-4
    hyperparameters = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'shuffle': True
    }
    train_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    train_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/train.tsv"
    validation_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    validation_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/dev.tsv"
    save_path = ""
    os.mkdir(save_path)
    with open(save_path+'hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    #Generators
    training_set = AudioDataset(train_data_folder, annotation_file)
    training_generator = torch.utils.data.DataLoader(training_set, **hyperparameters)

    validation_set = AudioDataset(train_data_folder, annotation_file)
    validation_generator = torch.utils.data.DataLoader(validation_set, **hyperparameters)

    print('LOAD TOKENIZER...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    print('INITIALIZING MODEL...')
    wav2vec_model = nn.DataParallel(ASR_CTC().cuda())

    print('TRAINING ALL LAYER...')
    optimizer = optim.AdamW(wav2vec_model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss()
    wav2vec_model, optimizer, best_val_loss = train(wav2vec_model, optimizer, criterion, epochs_all,
        data_loader_train, data_loader_valid, save_path)

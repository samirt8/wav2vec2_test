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


def validate(model, criterion, epoch, epochs, validation_generator, save_path, train_loss,
             best_val_loss, best_model_path):
    val_losses = []
    for inputs in tqdm(validation_generator, total=len(validation_generator)):
        with torch.no_grad():
            print("input : ", inputs["audio"])
            inputs, labels = inputs["audio"].cuda(), inputs["annotation"].cuda()
            output = model(inputs)
            output = output.view(-1, batch_size, nb_labels)
            print("output : ", output)

            output_length = torch.full(size=(batch_size,), fill_value=output.shape[0], dtype=torch.long)
            target_length = torch.randint(low=MAX_LEN-5, high=MAX_LEN, size=(batch_size,), dtype=torch.long)
            val_loss = criterion(output, labels, output_length, target_length)
            val_losses.append(val_loss.cpu().data.numpy())

            y_pred = torch.argmax(model(inputs), dim=-1).cpu().data.numpy().flatten()
            y_true = labels.cpu().data.numpy().flatten()

            print("Ground truth : ", tokenizer.decode(y_true))
            print("Prediction : ", tokenizer.decode(y_pred))

    val_loss = np.mean(val_losses)

    improved = ''

    model_path = save_path+'/model'
    torch.save(model.state_dict(), model_path)
    if val_loss < best_val_loss:
        improved = '*'
        best_val_loss = val_loss
        best_model_path = model_path


    progress_path = save_path+'/progress.csv'
    with open(progress_path, 'a') as f:
        f.write('{};{};{:.4f};{:.4f}\n'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch+1,
            train_loss,
            val_loss
            ))

    print("Epoch: {}/{}".format(epoch+1, epochs),
          "Loss: {:.4f}".format(train_loss),
          "Val Loss: {:.4f}".format(val_loss),
          improved)

    return best_val_loss, best_model_path



def train(model, optimizer, criterion, epochs, training_generator, validation_generator, save_path, best_val_loss=1e9):

    print_every = len(training_generator)
    best_model_path = None
    model.train()
    pbar = tqdm(total=print_every)

    for e in range(epochs):

        counter = 1

        for inputs in training_generator:

            inputs, labels = inputs["audio"].cuda(), inputs["annotation"].cuda()
            inputs.requires_grad = False
            labels.requires_grad = False
            output = model(inputs)
            output = output.view(-1, batch_size, nb_labels)
            output_length = torch.full(size=(batch_size,), fill_value=output.shape[0], dtype=torch.long)
            target_length = torch.randint(low=MAX_LEN-5, high=MAX_LEN, size=(batch_size,), dtype=torch.long)
            loss = criterion(output, labels, output_length, target_length)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss = loss.cpu().data.numpy()

            pbar.update()

            if counter % print_every == 0:

                pbar.close()
                model.eval()
                best_val_loss, best_model_path = validate(model, criterion, e, epochs, validation_generator, save_path, train_loss, best_val_loss, best_model_path)
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
    learning_rate = 3e-4
    MAX_LEN = 150
    hyperparameters = {
        'batch_size': batch_size,
        'shuffle': True
    }
    train_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    train_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/subset_train.tsv"
    validation_data_folder = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/clips"
    validation_annotation_file = "/media/nas/CORPUS_FINAL/Corpus_audio/Corpus_FR/COMMONVOICE/common-voice-fr/subset_dev.tsv"
    save_path = "/media/nas/samir-data/wav2vec2_models"
    #os.mkdir(save_path)
    with open(os.path.join(save_path,'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)

    #Generators
    training_set = AudioDataset(train_annotation_file, train_data_folder, MAX_LEN)
    training_generator = torch.utils.data.DataLoader(training_set, **hyperparameters)

    validation_set = AudioDataset(validation_annotation_file, validation_data_folder, MAX_LEN)
    validation_generator = torch.utils.data.DataLoader(validation_set, **hyperparameters)

    print('LOAD TOKENIZER...')
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    nb_labels = len(tokenizer.get_vocab())

    print('INITIALIZING MODEL...')
    wav2vec2_model = nn.DataParallel(ASR_CTC().cuda())

    print('TRAINING ALL LAYER...')
    optimizer = optim.AdamW(wav2vec2_model.parameters(), lr=learning_rate)
    criterion = nn.CTCLoss()
    wav2vec2_model, optimizer, best_val_loss = train(wav2vec2_model, optimizer, criterion, epochs,
        training_generator, validation_generator, save_path)

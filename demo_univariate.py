import torch
import argparse

import numpy as np
import pandas as pd
import torch.nn.functional as F

from mantis import Mantis
from mantis_trainer import MantisTrainer


def read_GestureMidAirD1():
    def read_data(file_name):
        data = pd.read_csv(file_name, sep='\t', header=None).to_numpy()
        X, y = torch.tensor(data[:, 1:], dtype=torch.float), data[:, 0]
        X = X.unsqueeze(-2)
        # reshape sequences to length equal to 512
        X = F.interpolate(X, size=512, mode='linear', align_corners=False)
        y -= 1
        X = X.numpy()
        y = y.astype(int)
        return X, y

    file_name_train = "data/GestureMidAirD1/GestureMidAirD1_TRAIN.tsv"
    file_name_test = "data/GestureMidAirD1/GestureMidAirD1_TEST.tsv"
    X_train, y_train = read_data(file_name_train)
    X_test, y_test = read_data(file_name_test)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuning_type", type=str, default='rf', help='rf: use FM to extract features, then learn RF on them, \
                                                                            head: fine-tune a linear head, \
                                                                            full: fine-tune the whole model, \
                                                                            scratch: same as full fine-tuning, but pre-trained model weights are not loaded.')
    parser.add_argument("--device", type=str, default='cuda', help='Device')
    args = parser.parse_args()
    fine_tuning_type = args.fine_tuning_type
    device = args.device
    if "cuda:" in device:
        gpu_id = int(device.split(':')[1])
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda', gpu_id)
    
    # ==== read data ====
    # your X_train and X_test should be of the shape (n_samples, 1, seq_len=512)
    # if original sequence length is different, resize it, for example, using the following function:
    # F.interpolate(X, size=512, mode='linear', align_corners=False)
    X_train, X_test, y_train, y_test = read_GestureMidAirD1()

    # ==== init the model, load the weights ====
    network = Mantis(device=device)
    if fine_tuning_type != 'scratch':
        network = network.from_pretrained("paris-noah/Mantis-8M")
    model = MantisTrainer(device=device, network=network)
    
    # ==== extract deep features, then random forest ====:
    if fine_tuning_type == 'rf':
        # get deep features
        Z_train = model.transform(X_train)
        Z_test = model.transform(X_test)
        # train a classifier
        from sklearn.ensemble import RandomForestClassifier
        predictor = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        predictor.fit(Z_train, y_train)
        # evaluate the performance
        y_pred = predictor.predict(Z_test)
        print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')
    # ==== using fit function of MantisTrainer ====
    else:
        # initialize some training parameters
        init_optimizer = lambda params: torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05)
        # fine-tune the model
        model.fit(X_train, y_train, num_epochs=100, fine_tuning_type=fine_tuning_type, init_optimizer=init_optimizer)
        # evaluate the performance
        y_pred = model.predict(X_test)
        print(f'Accuracy on the test set is {np.mean(y_test == y_pred)}')

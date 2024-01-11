import os
import pandas as pd
import numpy as np
import torch
from lgdcnn.train import Model
from lgdcnn.utils.get_compute_device import get_compute_device
from sklearn.metrics import roc_auc_score
compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)  
np.random.seed(RNG_SEED)


def get_model_matbench(Fork, mat_prop, i, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(Fork(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}{i}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'data/matbench_cv/{mat_prop}/train{i}.csv'
    val_data = f'data/matbench_cv/{mat_prop}/val{i}.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def get_model(mat_prop, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'data/benchmark_data/{mat_prop}/train.csv'
    val_data = f'data/benchmark_data/{mat_prop}/val.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=300, losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_matbench__predictions'
    # save_path = 'publication_predictions/onehot_matbench__predictions'
    # save_path = 'publication_predictions/random_200_matbench__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(Fork, lgdcnn_dir, model_name, mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    model = Model(Fork(compute_device=compute_device).to(compute_device),
                  model_name=f'{mat_prop}', verbose=verbose)
    model.load_network(model_name, f'{mat_prop}.pth') # multi_lstm_attention_residual_dpcnn_V8_512

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = os.path.join(lgdcnn_dir,"data","benchmark_data",mat_prop,file_name)
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9, train=False)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(Fork, lgdcnn_dir, model_name, mat_prop, classification, file_name, verbose=True):
    model = load_model(Fork, lgdcnn_dir, model_name, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'\n{mat_prop} ROC AUC: {auc:0.4f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'\n{mat_prop} mae: {mae:0.4g}')

    return model, mae
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, roc_auc_score
import torch
from torch.optim.lr_scheduler import CyclicLR
from lgdcnn.utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,EDM_CsvLoader, Scaler, DummyScaler, count_parameters)
from lgdcnn.utils.get_compute_device import get_compute_device
from lgdcnn.utils.optim import SWA

# To get the absolute path of the directory where the lgdcnn module is located
lgdcnn_dir = r"D:\deep\LGDCNN"

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

class Model():
    def __init__(self,
                 model,
                 model_name='None',
                 n_elements='infer',
                 capture_every=None,
                 verbose=True,
                 drop_unary=False,
                 scale=True):
        self.model = model
        self.model_name = model_name
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.n_elements = n_elements
        self.compute_device = model.compute_device
        self.fudge = 0.02  
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale
        if self.compute_device is None:
            self.compute_device = get_compute_device()
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

    def load_data(self, file_name, batch_size=2**7, train=False):
        self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(csv_data=file_name,
                                     batch_size=batch_size,
                                     n_elements=self.n_elements,
                                     inference=inference,
                                     verbose=self.verbose,
                                     drop_unary=self.drop_unary,
                                     scale=self.scale)
        self.n_elements = data_loaders.n_elements
        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.data[1]
        if train:
            self.train_len = len(y)
            self.scaler = DummyScaler(y) if self.classification else Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader


    def train(self):
        self.model.train()
        minima = []
        for _, data in enumerate(self.train_loader):
            X, y, _ = data
            y = self.scaler.scale(y)
            src, frac = X.squeeze(-1).chunk(2, dim=1)
            frac = frac * (1 + (torch.randn_like(frac))*self.fudge)  
            frac = torch.clamp(frac, 0, 1)
            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            src = src.to(self.compute_device,dtype=torch.long,non_blocking=True)
            frac = frac.to(self.compute_device,dtype=data_type_torch,non_blocking=True)
            y = y.to(self.compute_device,dtype=data_type_torch,non_blocking=True)

            if self.capture_every == 'step':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            output = self.model.forward(src, frac)
            prediction, uncertainty = output.chunk(2, dim=-1)
            loss = self.criterion(prediction.view(-1), uncertainty.view(-1), y.view(-1))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping: 
                self.lr_scheduler.step()

            swa_check = (self.epochs_step * self.swa_start - 1)
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)

        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            print(f'Epoch {self.epoch} failed to improve.')
            print(f'Discarded: {self.optimizer.discard_count}/{self.discard_n} weight updates')


    def fit(self, epochs=None, checkin=None, losscurve=False):
        assert_train_str = 'Please Load Training Data (self.train_loader)'
        assert_val_str = 'Please Load Validation Data (self.data_loader)'
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve['train'] = []
        self.loss_curve['val'] = []

        # change epochs_step
        self.epochs_step = 10

        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'stepping every {self.step_size} training passes,',
              f'cycling lr every {self.epochs_step} epochs')
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'running for {epochs} epochs')
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'checkin at {self.epochs_step*2} '
                  f'epochs to match lr scheduler')
        if epochs % (self.epochs_step * 2) != 0:
            updated_epochs = epochs
            epochs = updated_epochs

        self.step_count = 0
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer) 

        lr_scheduler = CyclicLR(self.optimizer,
                                base_lr=1e-4,
                                max_lr=6e-3,
                                cycle_momentum=False,
                                step_size_up=self.step_size) 
        self.lr_scheduler = lr_scheduler
        self.swa_start = 2  
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 5

        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            self.train()
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            if self.capture_every == 'epoch':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'
                print(epoch_str, train_str, val_str)

                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch+1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve['train'],
                             'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'],
                             's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

            if (epoch == epochs-1 or
                self.optimizer.discard_count >= self.discard_n):
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                tval = self.loss_curve['train']
                vval = self.loss_curve['val']
                df_loss = pd.DataFrame([xval, tval, vval]).T
                df_loss.columns = ['epoch', 'train loss', 'val loss']
                df_loss['swa'] = ['n'] * len(xval)
                df_loss.loc[df_loss['epoch'].isin(self.xswa), 'swa'] = 'y'
                # mk_path =  os.path.join(lgdcnn_dir,"train_results", "Benchmark","L-G-DCNN-TEST")
                # os.makedirs(mk_path, exist_ok=True)
                # df_loss.to_csv(os.path.join(mk_path, f'{self.model_name}_lc.csv'), index=False)

                # save output learning curve plot
                plt.figure(figsize=(8, 5))
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                plt.plot(xval, self.loss_curve['train'],
                         'o-', label='train_mae')
                plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                plt.title(f'{self.model_name}')
                plt.xlabel('epochs')
                plt.ylabel('MAE')
                plt.legend()
                # plt.savefig(os.path.join(mk_path, f'{self.model_name}_lc.png'))

            if self.optimizer.discard_count >= self.discard_n:
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates, '
                      f'early-stopping now 🙅🛑')
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    def predict(self, loader):
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0])/2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                if self.capture_flag:
                    self.formula_current = None
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                frac = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=data_type_torch,
                         non_blocking=True)
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)
                data_loc = slice(i*self.batch_size, i*self.batch_size+len(y), 1)
                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                formulae[data_loc] = formula
        self.model.train()
        return (act, pred, formulae, uncert)


    def save_network(self, name, model_name=None):
        if model_name is None:
            model_name = self.model_name
            mk_path =  os.path.join(lgdcnn_dir,"models", "Benchmark",name)
            os.makedirs(mk_path, exist_ok=True)
            path  = os.path.join(lgdcnn_dir,"models", "Benchmark", name, f'{model_name}.pth') 
            print(f'Saving network ({model_name}) to {path}')
        else:
            path  = os.path.join(lgdcnn_dir,"models", "Benchmark", name, f'{model_name}.pth') 
            print(f'Saving checkpoint ({model_name}) to {path}')

        save_dict = {'weights': self.model.state_dict(),
                     'scaler_state': self.scaler.state_dict(),
                     'model_name': model_name}
        torch.save(save_dict, path)

    def load_network(self, name, path):
        """
        Load a network from a given path.

        Parameters:
            path (str): The path to the network file.

        Returns:
            None
        """
        path  = os.path.join(lgdcnn_dir,"models", "Benchmark", name, path) 
        network = torch.load(path, map_location=self.compute_device)
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(network['weights'])
        self.scaler.load_state_dict(network['scaler_state'])
        self.model_name = network['model_name']



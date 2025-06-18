from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, iTransformer, Crossformer, FEDformer, DLinear, Linear, NLinear, PatchTST, Naive, ES, TimeMixer
from models.nbeats_models import interpretable, generic 
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import shutil

warnings.filterwarnings('ignore')

from autoaugment.loggers import CheckpointLogger



class Exp_Main(Exp_Basic):
    def __init__(self, args, checkpoint_logger=None, augment_ts=None, use_early_stopping=True):
        super(Exp_Main, self).__init__(args)
        if args.use_logger:
            self.checkpoint_logger = checkpoint_logger
            if self.checkpoint_logger is None:
                self.checkpoint_logger = CheckpointLogger(args)
        else:
            self.checkpoint_logger = None
        self.augment_ts = augment_ts
        self.use_early_stopping = use_early_stopping

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'NBEATS_G': generic,
            'NBEATS_I': interpretable, 
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST, 
            'Naive': Naive,
            'ES': ES,  
            'TimeMixer': TimeMixer,
            'iTransformer': iTransformer,
            'Crossformer': Crossformer
        }

        if self.args.model == 'NBEATS_G':
            model = model_dict[self.args.model](input_size=self.args.seq_len,
                                               output_size=self.args.pred_len,
                                               stacks=self.args.nb_stacks,
                                               layers=self.args.nb_layers,
                                               layer_size=self.args.nb_layer_size).float()
        elif self.args.model == 'NBEATS_I':
            model = model_dict[self.args.model](input_size=self.args.seq_len,
                                               output_size=self.args.pred_len,
                                               trend_blocks=self.args.nb_trend_blocks,
                                               trend_layers=self.args.nb_trend_layers,
                                               trend_layer_size=self.args.nb_trend_layer_size,
                                               seasonality_blocks=self.args.nb_seasonality_blocks,
                                               seasonality_layers=self.args.nb_seasonality_layers,
                                               seasonality_layer_size=self.args.nb_seasonality_layer_size,
                                               num_of_harmonics=self.args.nb_num_of_harmonics,
                                               degree_of_polynomial=self.args.nb_degree_of_polynomial).float()
        else:                                        
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, generator=None, kfold=1, kfold_i=1, kfold_flag='train'):
        data_set, data_loader = data_provider(self.args, flag, generator, kfold, kfold_i, kfold_flag)
        return data_set, data_loader
    
    def _set_ts_augment(self, augment_ts):
        self.augment_ts = augment_ts

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion=None, augment=False):
        if criterion is None:
            criterion = self._select_criterion()
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float().to(self.device)
                batch_x_mark = batch[2].float().to(self.device)
                batch_y_mark = batch[3].float().to(self.device)

                if len(batch) == 6:
                    batch_trend = batch[4].float().to(self.device)
                    batch_seasonality = batch[5].float().to(self.device)
                else:
                    batch_trend = None
                    batch_seasonality = None

                if (self.args.data in ['PEMS', 'Solar']) and self.args.model == 'TimeMixer':
                    batch_x_mark = None
                    batch_y_mark = None

                if self.augment_ts is not None and augment:
                    batch_augment = {"data": torch.cat([batch_x, batch_y[:, self.args.label_len:]], axis=1),
                                    "trend": batch_trend,
                                    "seasonality": batch_seasonality}
                    x_len = batch_x.shape[1]
                    y_len = batch_y.shape[1]

                    aug_output = self.augment_ts.augment(batch_augment, horizon=self.args.pred_len)
                    batch_x = aug_output[:, :x_len]
                    batch_y = aug_output[:, -y_len:]

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)               
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = self.model(batch_x, input_mask)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, stage2=False, save=False, kfold=1, kfold_i=1):
        loader_generator = torch.Generator()
        if self.args.method == "ensemble" and stage2:
            run_seed = self.args.ensemble_seed + self.args.random_seed_aa
            loader_generator.manual_seed(run_seed)

        train_data, train_loader = self._get_data(flag='train', generator=loader_generator, kfold=kfold, kfold_i=kfold_i)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        start_time = time.time()
        time_now = start_time
        train_steps = len(train_loader)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                               steps_per_epoch=train_steps,
                                               pct_start=self.args.pct_start,
                                               epochs=self.args.train_epochs,
                                               max_lr=self.args.learning_rate)
        else:
            scheduler = None

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if stage2 == False:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
            if self.checkpoint_logger is not None:
                self.checkpoint_logger.create_checkpoint(model=self.model,
                                                       model_optim=model_optim,
                                                       epoch=0,
                                                       early_stopping=early_stopping,
                                                       scheduler=scheduler,
                                                       loader_generator=loader_generator)

            max_epochs = self.args.train_epochs
            epochs_range = range(max_epochs)
            file_name = 'checkpoint.pth'
            if self.args.method == "ensemble":
                file_name = f'checkpoint_ensemble_lookback{self.args.seq_len}_{self.args.ensemble_seed}.pth'
        else:
            self.model, model_optim, base_epoch, max_epochs, es, scheduler = self.checkpoint_logger.load_checkpoint(
                model=self.model,
                model_optim=model_optim,
                scheduler=scheduler,
                loader_generator=loader_generator
            )
            file_name = 'checkpoint_aug.pth'
            if self.args.method == "ensemble":
                file_name = f'checkpoint_ensemble_lookback{self.args.seq_len}_{self.args.ensemble_seed}.pth'

            if self.use_early_stopping:
                max_epochs = self.args.train_epochs
                early_stopping = es
                val_loss = np.inf if len(self.checkpoint_logger.val_loss) == 0 else self.checkpoint_logger.val_loss[-1]
                early_stopping.save_checkpoint(val_loss, self.model, path, file_name)
            else:
                early_stopping = None
            epochs_range = range(base_epoch, max_epochs)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in epochs_range:
            iter_count = 0
            train_loss = []    
            self.model.train()        
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch[0].float().to(self.device)
                batch_y = batch[1].float().to(self.device)
                batch_x_mark = batch[2].float().to(self.device)
                batch_y_mark = batch[3].float().to(self.device)

                if len(batch) == 6:
                    batch_trend = batch[4].float().to(self.device)
                    batch_seasonality = batch[5].float().to(self.device)
                else:
                    batch_trend = None
                    batch_seasonality = None
                
                if (self.args.task_id in ['PEMS', 'Solar']) and self.args.model == 'TimeMixer':
                    batch_x_mark = None
                    batch_y_mark = None

                if self.args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    dec_inp = None

                if self.augment_ts is not None:
                    batch_augment = {"data": torch.cat([batch_x, batch_y[:, self.args.label_len:]], axis=1),
                                   "trend": batch_trend,
                                   "seasonality": batch_seasonality}
                    x_len = batch_x.shape[1]
                    y_len = batch_y.shape[1]

                    aug_output = self.augment_ts.augment(batch_augment, horizon=self.args.pred_len)
                    batch_x = aug_output[:, :x_len]
                    batch_y = aug_output[:, -y_len:]
                
                if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = self.model(batch_x, input_mask)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    if self.args.use_amp:                            
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                outputs = self.model(batch_x)
                            else:            
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                            outputs = self.model(batch_x) 
                        else:                           
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            if early_stopping is not None:
                early_stopping(vali_loss, self.model, path, epoch + 1, file_name)
                break_run = early_stopping.early_stop
            else:
                break_run = False

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            
            if self.checkpoint_logger is not None:
                self.checkpoint_logger.create_checkpoint(
                    self.model,
                    model_optim,
                    epoch + 1,
                    train_loss,
                    vali_loss,
                    test_loss,
                    early_stopping,
                    scheduler,
                    loader_generator=loader_generator
                )

            if break_run:
                if self.checkpoint_logger is not None:
                    self.checkpoint_logger.finalize_training(early_stopping.best_epoch, start_time)
                break
            
        if early_stopping is not None and not stage2:
            best_model_path = path + '/' + file_name
            self.model.load_state_dict(torch.load(best_model_path))
            if not early_stopping.early_stop and self.checkpoint_logger is not None:
                self.checkpoint_logger.finalize_training(early_stopping.best_epoch, start_time)

        is_empty = not os.listdir(path)
        if stage2 and save:
            torch.save(self.model.state_dict(), path + '/' + file_name)
        elif (not stage2 and not save) or (is_empty and not save):
            shutil.rmtree(path)

        return self.model

    def test(self, setting, test=0, autoaugment=False, save=False, file_name='checkpoint.pth'):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, file_name)))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        aa = "aa_" if autoaugment else ""

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = self.model(batch_x, input_mask)
                else:
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                outputs = self.model(batch_x)
                            else:    
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                        input = np.expand_dims(input, -1)
                        true = np.expand_dims(true, -1)
                        pred = np.expand_dims(pred, -1)
                    
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, aa + str(i) + '.pdf'))

                    np.save(os.path.join(folder_path, aa + str(i) + '_true.npy'), true[0, :, -1])
                    np.save(os.path.join(folder_path, aa + str(i) + '_input.npy'), input[0, :, -1])
                    np.save(os.path.join(folder_path, aa + str(i) + '_pred.npy'), pred[0, :, -1])

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        if save:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')
            f.close()
    
            np.save(folder_path + aa + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + aa + 'pred.npy', preds)
            np.save(folder_path + aa + 'true.npy', trues)

        return {'mae': mae, 'mse': mse}

    def predict(self, setting, load=False, file_name='checkpoint.pth'):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + file_name
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                    batch_x = batch_x.squeeze(-1)
                    batch_y = batch_y[:, -self.args.pred_len:].squeeze(-1)              
                    input_mask = torch.ones(batch_x.shape).to(self.device)
                    outputs = self.model(batch_x, input_mask)
                else:
                    dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

    def test_ensemble(self, setting, test=0, autoaugment=False, save=False, file_name='checkpoint.pth'):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        model_paths_list = os.listdir(path)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        aa = "aa_" if autoaugment else ""

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_preds = []
                
                for m_path in model_paths_list:
                    m_path = os.path.join(path, m_path)
                    self.model.load_state_dict(torch.load(m_path))
                    self.model.eval()
                    
                    if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                        batch_x_curr = batch_x.squeeze(-1)
                        batch_y_curr = batch_y[:, -self.args.pred_len:].squeeze(-1)
                        input_mask = torch.ones(batch_x_curr.shape).to(self.device)
                        outputs = self.model(batch_x_curr, input_mask)
                    else:
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                    outputs = self.model(batch_x)
                                else:    
                                    if self.args.output_attention:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                    else:
                                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if self.args.model in ['Naive', 'PatchTST', 'DLinear', 'NLinear', 'Linear', 'ES']:
                                outputs = self.model(batch_x)
                            else:
                                if self.args.output_attention:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    outputs = outputs.detach().cpu().numpy()
                    batch_preds.append(outputs)

                pred = np.median(batch_preds, axis=0)
                batch_y = batch_y.detach().cpu().numpy()
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if self.args.model in ['NBEATS_G', 'NBEATS_I']:
                        input = np.expand_dims(input, -1)
                        true = np.expand_dims(true, -1)
                        pred = np.expand_dims(pred, -1)
                    
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, aa + str(i) + '.pdf'))

                    np.save(os.path.join(folder_path, aa + str(i) + '_true.npy'), true[0, :, -1])
                    np.save(os.path.join(folder_path, aa + str(i) + '_input.npy'), input[0, :, -1])
                    np.save(os.path.join(folder_path, aa + str(i) + '_pred.npy'), pred[0, :, -1])

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        if save:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n\n')
            f.close()
    
            np.save(folder_path + aa + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + aa + 'pred.npy', preds)
            np.save(folder_path + aa + 'true.npy', trues)

        return {'mae': mae, 'mse': mse}
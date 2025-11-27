import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

    # Builds the autoformer model and gets the data using data loader

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# from models import Informer, Autoformer, Transformer, Reformer
from models import Autoformer, TSMixer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.external_loader = getattr(args, "data_loader", None)


    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            # 'Informer': Informer,
            # 'Transformer': Transformer,
            'TSMixer': TSMixer,
        }
        if hasattr(model_dict[self.args.model], "Model"):
            model = model_dict[self.args.model].Model(self.args).float()
        else:
            model = model_dict[self.args.model](self.args).float()
        return model

    def _get_data(self, flag):
        # If a custom multi-dataset loader was provided externally, use it
        if self.external_loader is not None:
            print(f"✅ Using external data loader for {flag} data (multi-dataset).")
            # We still need to return something as data_set, so we use None
            return None, self.external_loader
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    @torch.no_grad()
    def cache_teacher_preds(self, setting, flag='train', save_path=None):
        """
        Run the trained Autoformer on a dataset split and save predictions + ground truth for KD.
        """
        data_set, data_loader = self._get_data(flag=flag)
        self.model.eval()

        preds, trues = [], []
        for i, batch in enumerate(data_loader):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                dataset_id = None

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            outputs, batch_y_true = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(outputs.detach().cpu())
            trues.append(batch_y_true.detach().cpu())

        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        if save_path is None:
            save_path = f'./cache/{setting}_{flag}_teacher.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'teacher_preds': preds, 'y_true': trues}, save_path)
        print(f"[KD] Saved teacher predictions for {flag} → {save_path}")
        return save_path

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if (i + 1) % 100 == 0:
                    print(f"\r    [Val] Processed {i+1}/{len(vali_loader)} batches...", end="", flush=True)

                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    dataset_id = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if outputs.shape[-1] != batch_y.shape[-1]:
                      batch_y = batch_y[:, :, :outputs.shape[-1]]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        print(f"\n    [Val] Completed {len(vali_loader)} batches.")
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        print(f"🧮 Train loader has {len(train_loader)} batches per epoch")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                # unpack batch
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    dataset_id = None

                iter_count += 1
                global_step = epoch * len(train_loader) + i  # ✅ running global iteration counter

                # === NEW: early exit after fixed number of total iterations ===
                if getattr(self.args, "max_train_steps", None) is not None:
                    if global_step >= self.args.max_train_steps:
                        print(f"\n🛑 Reached max_train_steps={self.args.max_train_steps}, stopping early.")
                        early_stopping.early_stop = True
                        break
                # =============================================================

                # standard training code
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if outputs.shape[-1] != batch_y.shape[-1]:
                      batch_y = batch_y[:, :, :outputs.shape[-1]]

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


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    dataset_id = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if outputs.shape[-1] != batch_y.shape[-1]:
                    batch_y = batch_y[:, :, :outputs.shape[-1]]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

                # ✅ Also save results to checkpoints folder if available (for external evaluation)
        try:
            save_dir = getattr(self.args, "checkpoints", None)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, "test_results.npy"), preds)
                np.save(os.path.join(save_dir, "true_results.npy"), trues)
                print(f"[Info] Copied prediction results to {save_dir}")
        except Exception as e:
            print(f"[Warning] Could not copy results to checkpoints folder: {e}")


        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                if len(batch) == 5:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                    dataset_id = None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

   
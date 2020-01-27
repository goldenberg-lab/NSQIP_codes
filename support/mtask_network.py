import numpy as np
import pandas as pd
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from support.support_funs import stopifnot
from support.mdl_funs import normalize
from support.mdl_funs import idx_iter

# def multistrat_split(Y, prop=0.1):
#     stopifnot(isinstance(Y,pd.DataFrame))
#     n = Y.shape[0]
#     yidx = [np.where(Y.iloc[:,cc]==1)[0] for cc in range(Y.shape[1])]
#     lidx = pd.Series([len(z) for z in yidx])
#     nidx = np.round(lidx*prop).astype(int)
#     nidx = np.where(nidx == 0, 1, nidx)
#     nprop = int(np.round(n * prop))

def sigmoid(x):
    return (1/(1+np.exp(-x)))

# self = mtask_nn()
# data=Xtrain;lbls=Ytrain;nepochs=10;mbatch=1000;val_prop=0.1;lr=0.001;ii=0;jj=0
class mtask_nn(nn.Module):
    def __init__(self):
        super(mtask_nn, self).__init__()

    def architecture(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        # -- define architecture -- #
        self.fc1 = nn.Linear(self.n_input, 100)
        #self.bn1 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 50)
        #self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 25)
        #self.bn3 = nn.BatchNorm1d(25)
        self.fc4 = nn.Linear(25, self.n_output)

    def forward(self,x):
        x = F.leaky_relu(self.fc1(x)) #self.bn1()
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        return(x)

    def transform(self,data,check=True):
        return( torch.Tensor(self.enc.transform(data, check=check)) )

    def predict(self,data,check=True):
        with torch.no_grad():
            return( self.eval()(self.transform(data,check)).numpy() )

    def fit(self, data, lbls, nepochs=100, mbatch=1000, val_prop=0.1, lr=0.001):
        n = data.shape[0]
        stopifnot(n == lbls.shape[0])
        if len(lbls.shape) == 1:
            lbls = lbls.reshape([n, 1])
        k = lbls.shape[1]
        check, rr = True, 0
        while check:
            rr += 1
            idx_train, idx_val = train_test_split(np.arange(n), test_size=val_prop, random_state=rr)
            check = not all(lbls.iloc[idx_val].apply(lambda x: x[~(x==-1)].sum(),axis=0) > 0)
        n_train, n_val = len(idx_train), len(idx_val)
        # Find encodings/normalization
        self.enc = normalize(copy=False)
        self.enc.fit(data.loc[idx_train])
        Yval = lbls.iloc[idx_val].values # Pre-compute for faster eval
        nY_val= np.apply_along_axis(func1d=lambda x: x[~(x==-1)].sum(),axis=0,arr=Yval)
        nY_train = lbls.iloc[idx_train].apply(lambda x: x[~(x == -1)].sum(),0).values
        wY_train = (n_train / nY_train - 1).reshape([1,k])
        # Define architecture
        torch.manual_seed(1234)
        self.architecture(n_input=self.enc.p2, n_output=k)
        # Create loss function (note we do not set class because weights will be iterative)
        loss_fun = nn.BCEWithLogitsLoss
        # Set up optimizer
        optimizer = torch.optim.Adagrad(params=self.parameters(), lr=lr)
        self.res = []
        nll_epoch = []
        tstart = time.time()
        for ii in range(nepochs):
            idx_batches = idx_iter(n_train, mbatch, ii)
            nbatch = len(idx_batches)
            print('---- Epoch %i of %i ----' % (ii+1, nepochs))
            nll_batch = []
            nll_batch_cc = []
            for jj in range(nbatch):
                # if (jj+1)%10==0:
                #     print('Batch %i of %i' % (jj+1, nbatch))
                idx_jj = idx_train[idx_batches[jj]]
                optimizer.zero_grad()
                # --- Forward pass --- #
                out_jj = self.forward(self.transform(data.iloc[idx_jj],False))
                Y_jj = lbls.iloc[idx_jj].values
                W_jj = torch.Tensor(np.where(Y_jj == -1, 0, 1) * ((Y_jj * wY_train)+1))
                Y_jj = torch.Tensor(Y_jj)
                loss_jj = loss_fun(reduction='mean',weight=W_jj)(input=out_jj,target=Y_jj)
                # --- Backward pass --- #
                loss_jj.backward()
                optimizer.step()
                nll_batch.append(loss_jj.item())
                with torch.no_grad():
                    loss_jj_cc = loss_fun(reduction='none', weight=W_jj)(input=out_jj, target=Y_jj).mean(axis=0).numpy()
                    nll_batch_cc.append(loss_jj_cc)
            nll_epoch.append(np.mean(nll_batch))
            #df_nll = pd.DataFrame({'cn':lbls.columns,'y1':nY_train,'nll':np.vstack(nll_batch_cc).mean(axis=0)})
            if (ii+1) % 10 == 0:
                # Check gradient stability
                for layer, param in self.named_parameters():
                    print('layer: %s, std: %0.4f' % (layer, param.grad.std().item()))
                # Check for early stpoping
                phat_val = self.predict(data.iloc[idx_val])
                holder = []
                for cc in range(Yval.shape[1]):
                    idx_cc = ~(Yval[:, cc] == -1) #
                    act_cc = Yval[idx_cc,cc] #
                    pred_cc = sigmoid(phat_val[idx_cc, cc])
                    holder.append([roc_auc_score(act_cc,pred_cc),
                                 average_precision_score(act_cc, pred_cc),
                                 log_loss(act_cc,pred_cc)])
                res_ii = pd.DataFrame(np.vstack(holder),index=lbls.columns,
                                      columns=['auc','ppv','nll']).reset_index()
                res_ii = pd.concat([pd.DataFrame({'iter': ii + 1, 'n': nY_val}), res_ii], 1)
                if ii > 10:
                    self.res = pd.concat([self.res, res_ii],0).reset_index(drop=True)
                    val_score = self.res.drop(columns=['ppv', 'nll']).rename(columns={'index': 'cc'})
                    val_score = val_score.sort_values(['cc', 'iter']).reset_index(drop=True)
                    val_score['d_auc'] = (val_score.auc - val_score.groupby('cc').auc.shift(+1)) * val_score.n
                    val_score = val_score.groupby('iter').d_auc.mean().reset_index().fillna(0)
                    if not all(val_score.d_auc >= 0):
                        print('#### EARLY STOPPING AT ITERATION %i ####' % (ii+1) )
                        break
                    else:
                        print(val_score)
                else:
                    self.res = res_ii
                self.res
        tend = time.time()

# qq = pd.DataFrame({'y':act_cc.astype(str),'p':pred_cc})
# g = sns.FacetGrid(data=qq,hue='y')
# g.map(sns.distplot,'p')
# g.add_legend()

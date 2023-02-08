
# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split


sns.set_style('darkgrid')
plt.style.use("fivethirtyeight")

import preprocess
import Analysis

x_names = ["MoneFd","AdvPay","Invtr","LTEquInv","IntanAss","STLoan","AccPay","SalPay","TaxPay","ShrCap","CapSur",
           "SurRes","RetEar_x","MOpeRev","DisctAlw","MOpeCost","OpeTaxSur","MOpePrf","OthOpePrf","OpeExp",
           "AdmExp","FinExp1","OthExp","OpePrf","InvIncm","FutGain","SubsdRev","NOpeRev","NOpeExp","IncmTax","FinRefd",
           "MinInt1","UncfInvLos1","MAPrf","RetEarBegYr","RedPrfCap","SpoPrf","SurResCapSur","SprForRetEar","DistPrf",
           "AppSurRes","AppWelfFd","StafFd","ForCapRes","AppResFd","AppExpFd","RetInv","DistPrfSH","DivPPrefStk",
           "AppEarSur","DivPComStk","ComStkDivConverShrCap","AccLos","RetEary","PrfDispoDepart","LosNatDist",
           "NetPrfChgAccPol","OthSpeItem","GoodServR","RentR","VATRrefd","OthRefdTD","RefdTD","OthROpe","CIFOpe",
           "GoodServP","OpeLeaP","StafP","VATP","IncmTaxP","OthTaxP","AllTaxP","OthPOpe","COFOpe","InvRetR","DivPrfR",
           "BdIntR","DispoAssR","OthInvR","CIFInv","AcquiAssP","EquInvP","DbInvP","OthPInv","COFInv","EquInvR",
           "SubsdMinInvR","BdIssR","LoaR","OthRfIn","CIFfIn","BorwP","FdExpP","DivPrfP","SubsdMinP","IntExpP",
           "FinLeaP","RedRegCapP","SubsdMinLegCapDecrP","OthPFin","COFFin","EffFERCCE","LiarepFixAss","LiaRepInv",
           "LTInvFixAss","LiaRepInvtr","FixAssFin","MinInt2","MaPrf","UncfInvLos2","ProvBadDb","DeprFixAss",
           "AmotIntanAss","AmotPropExp","AmotLTDefChr","DecrDefChr","IncrAccrExp","LosDispoAss","LosretireFixAss",
           "FinExp2","InvLos","DefTax","DecrInvtr","InvtrLos","DecrOpeRecv","IncrOpePay","Oth","EndPdCas","BegPdCas",
           "CCEEndPd","CCEBegPd","BreakMark","IfBreak"

]

class GetDataset(object):
    def __init__(self, df):
        super(GetDataset, self).__init__()
        self.df = df
        self.company = []
        for c,d in df.index.values:
            if c not in self.company:
                self.company.append(c)

        self.df["WillBreak"] = df["IfBreak"]
        self.datad = {}
        data_all=[]
        for k in self.company:
            c_data = self.df.loc[k]
            c_data["WillBreak"] = c_data["WillBreak"].shift(-1)
            c_data = c_data.dropna()
            if len(c_data.index)>1:
                data_all.append(c_data)
                self.datad[k] = c_data

        self.df = pd.concat(data_all, ignore_index=True)


    def get_dataset(self, scale=True, stationary=False, indicators=False):
        '''
            Input: scale - if to scale the input data
        '''
        transfoms = {}
        self.y_scaler = None
        if scale=="MinMaxScaler":
            self.y_scaler = MinMaxScaler()
            for f in x_names:
                transfoms[f] = MinMaxScaler().fit(self.df[f].values.reshape(-1,1))

        if len(transfoms.items())>0:
            for k,v in self.datad.items():
                for f in x_names:
                    v[f] = transfoms[f].transform(v[f].values.reshape(-1,1))



    def get_size(self):
        '''
            Output: returns the length of the dataset
        '''
        return len(self.x_data)


    def split(self, train_split_ratio=0.8, time_period=30):
        '''
            Input: train_split_ratio - percentage of dataset to be used for
                                       the training data (float)
                   time_period - time span in days to be predicted (in)

            Output: lists of the training and validation data (input values and target values)
                    size of the training data
        '''
        x_period_data=[]
        y_period_data=[]
        for k,v in self.datad.items():
            row_num = len(v.index)
            if row_num<time_period:
                continue
            x_data = v[x_names].values
            y_data = v["WillBreak"].values
            x_period_data.append(np.asarray([x_data[i-time_period:i] for i in range(time_period, row_num+1)]))
            y_period_data.append(np.asarray(y_data[time_period-1:]))

        x_period_data = np.concatenate(x_period_data)
        y_period_data = np.concatenate(y_period_data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_period_data,y_period_data,test_size=0.2)

        train_data_size = len(self.x_train)
        return [self.x_train, self.y_train], [self.x_test, self.y_test], train_data_size


    def get_torchdata(self):
        self.x_train_tensor = torch.from_numpy(self.x_train).type(torch.Tensor)
        self.x_test_tensor = torch.from_numpy(self.x_test).type(torch.Tensor)

        self.y_train_tensor = torch.from_numpy(self.y_train).type(torch.Tensor)
        self.y_test_tensor = torch.from_numpy(self.y_test).type(torch.Tensor)

        return [self.x_train_tensor, self.y_train_tensor], [self.x_test_tensor, self.y_test_tensor]


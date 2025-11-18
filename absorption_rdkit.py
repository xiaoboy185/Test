import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import ast
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# One-hot Encoding

from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
import ast
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# One-hot Encoding

from rdkit import Chem
from rdkit.Chem import rdchem

def parse_features(feature_str):
    # 将字符串转换为嵌套列表
    feature_list = ast.literal_eval(feature_str)
    # 将嵌套列表转换为 NumPy 数组
    feature_array = np.array(feature_list)
    return feature_array

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def subgraph_features(subgraph, use_chirality=False):
    """提取子图的特征"""
    atoms = subgraph.GetAtoms()
    node_f = [atom_features(atom, use_chirality=use_chirality) for atom in atoms]
    edge_index = get_bond_pair(subgraph)
    edge_attr = [bond_features(bond, use_chirality=use_chirality) for bond in subgraph.GetBonds()]
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float))
    return data

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

import torch
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','S','F','Si','P','Cl','Br',
                                                                   'Na','I','B','H', 'Se', 'Sn', 'Te'
                                                                   'Ge']) + 
                           one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5]) + 
                           one_of_k_encoding_unk(atom.GetValence(rdchem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4]) + 
                           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + 
                           one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, 
                                                                           Chem.rdchem.HybridizationType.SP2, 
                                                                           Chem.rdchem.HybridizationType.SP3,]) + 
                           [atom.GetIsAromatic()])
    if not explicit_H:
        results = np.array(results.tolist() + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4]))
    if use_chirality:
        try:
            results = np.array(results.tolist() + 
                               one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + 
                               [atom.HasProp('_ChiralityPossible')])
        except:
            results = np.array(results.tolist() + 
                               [False, False] + 
                               [atom.HasProp('_ChiralityPossible')])

    return np.array(results)

# Edge Feature Matrix
def bond_features(bond, use_chirality=False):
#    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),
                  bond.IsInRing()]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()),
                                                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

# Edge Index List
def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors

def calculate_molecular_features(mol):
    """计算RDKit分子特征"""
    features = [
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NOCount(mol) / float(mol.GetNumHeavyAtoms()),
        len(mol.GetAromaticAtoms()) / float(mol.GetNumHeavyAtoms()),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        Descriptors.FractionCSP3(mol),
    ]
    molecular_features_tensor = torch.tensor(features, dtype=torch.float)
    molecular_features_array = molecular_features_tensor.numpy()
    molecular_features_array = molecular_features_array.reshape(1, -1)
    return molecular_features_array

# Graph
def mol2vec(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features(atom) for atom in atoms]
    
    # 计算RDKit特征
    global_features_rdkit = calculate_molecular_features(mol)
    
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
    
    for bond in bonds:
        edge_attr.append(bond_features(bond))
    
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    
    # 存储RDKit特征
    data.global_features_rdkit = torch.tensor(global_features_rdkit, dtype=torch.float)

    return data

def make_regre_mol(df): 
    mols1_key = []
    mols2_key = []
    mols_value = []

    for i in range(df.shape[0]):
        
        mols1_key.append(Chem.MolFromSmiles(df['Chromophore'].iloc[i]))
        
        mols2_key.append(Chem.MolFromSmiles(df['Solvent'].iloc[i]))
        mols_value.append(df['Absorption max (nm)'].iloc[i])
    return mols1_key, mols2_key, mols_value

def make_regre_vec(mols1, mols2, value, value_orig):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(mols1)):
        m1 = mols1[i]
        m2 = mols2[i]
        y = value[i]

        v1 = mol2vec(m1)
        v2 = mol2vec(m2)

        X1.append(v1)
        X2.append(v2)
        Y.append(y)

    for i, data in enumerate(X1):
        y = Y[i]
        y_orig = value_orig[i]  # 原始值
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)  # 原始目标值
    for i, data in enumerate(X2):
        y = Y[i]
        y_orig = value_orig[i]
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)
    return X1, X2

def make_regre_vec_test(mols1, mols2, value, value_orig):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(mols1)):
        m1 = mols1[i]
        m2 = mols2[i]
        y = value[i]

        v1 = mol2vec(m1)
        v2 = mol2vec(m2)

        X1.append(v1)
        X2.append(v2)
        Y.append(y)

    for i, data in enumerate(X1):
        y = Y[i]
        y_orig = value_orig[i]  # 原始值
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)  # 原始目标值
    for i, data in enumerate(X2):
        y = Y[i]
        y_orig = value_orig[i]
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)
    return X1, X2

def make_regre_vec_val(mols1, mols2, value, value_orig):
    X1 = []
    X2 = []
    Y = []
    for i in range(len(mols1)):
        m1 = mols1[i]
        m2 = mols2[i]
        y = value[i]

        v1 = mol2vec(m1)
        v2 = mol2vec(m2)

        X1.append(v1)
        X2.append(v2)
        Y.append(y)

    for i, data in enumerate(X1):
        y = Y[i]
        y_orig = value_orig[i]  # 原始值
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)  # 原始目标值
    for i, data in enumerate(X2):
        y = Y[i]
        y_orig = value_orig[i]
        data.y = torch.tensor([y], dtype=torch.float)
        data.y_orig = torch.tensor([y_orig], dtype=torch.float)
    return X1, X2
#### Example
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool

# Classification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ARMAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool, global_mean_pool

# Classification
n_features=38
conv_dim1 = 64
conv_dim2 = 64
conv_dim3 = 64
concat_dim = 32
pred_dim1 = 64
pred_dim2 = 64
pred_dim3 = 64
out_dim = 1
global_feature_dim = 32
head=8
edge_features=8
dropout=0.33
head=8
edge_features=8
dropout=0.33
from torch_geometric.nn import GATConv, GPSConv

class GATlayer(nn.Module):
    def __init__(self, n_features, edge_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout):
        super(GATlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.concat_dim = concat_dim
        self.dropout = dropout
        self.head = head
        # 使用 GATConv 层
        self.conv1 = GATConv(self.n_features, self.conv_dim1, self.head, concat=True)
        self.bn1 = BatchNorm1d(self.conv_dim1 * self.head)
        self.residual1 = nn.Sequential(
            Linear(n_features, conv_dim1 * head),  # 残差路径维度匹配
            BatchNorm1d(conv_dim1 * head)
        )
     
        self.conv2 = GATConv(self.conv_dim1 * self.head, self.conv_dim2, self.head, concat=True)
        self.bn2 = BatchNorm1d(self.conv_dim2 * self.head)
        self.residual2 = nn.Sequential(
            Linear(conv_dim1 * head, conv_dim2 * head),
            BatchNorm1d(conv_dim2 * head))
        
        self.conv3 = GATConv(self.conv_dim2 * self.head, self.conv_dim3, self.head, concat=True)
        self.bn3 = BatchNorm1d(self.conv_dim3 * self.head)
        self.residual3 = nn.Sequential(
            Linear(conv_dim2 * head, conv_dim3 * head),
            BatchNorm1d(conv_dim3 * head))
        
        self.conv4 = GATConv(self.conv_dim3 * self.head, self.concat_dim, self.head, concat=True)
        self.bn4 = BatchNorm1d(self.concat_dim * self.head)
       # self.gcn = GCNConv(self.n_features, self.conv_dim1, cached=False)
 #attention
        self.attention = nn.Sequential(
            nn.Linear(self.concat_dim * self.head * 2, self.concat_dim * self.head),
            nn.ReLU(),
            nn.Linear(self.concat_dim * head, 1),
            nn.Sigmoid())

    def forward(self, data, device):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        # 分别获取RDKit特征
        global_features_rdkit = data.global_features_rdkit.to(device)
        
        #x = self.gcn(xa, edge_index)
        original_x = x
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)
    
        res1 = self.residual1(original_x)     #
        x1 = x1 + res1  # 添加残差连接


        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x2 = self.bn2(x2)
        res2 = self.residual2(x1)     #
        x2 = x2 + res2  # 添加残差连接
        
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        x3 = self.bn3(x3)
        res3 = self.residual3(x2)     #
        x3 = x3 + res3  # 添加残差连接
        
        x = F.relu(self.conv4(x3, edge_index, edge_attr))
        x = self.bn4(x)

 #       sub_features = sub.to(device)
 #       attention_weights = self.attention(torch.cat((x, sub_features), dim=1))
 #       x = x * attention_weights + sub_features * (1 - attention_weights)

    
        #x = torch.cat((x, xa, sub), dim=1)
        # 获取全局特征
        x = global_add_pool(x, data.batch)
        # 将全局特征拼接到图特征中
        x = torch.cat((x, global_features_rdkit), dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class FClayer1(nn.Module):
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout):
        super(FClayer1, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.out_dim = out_dim
        self.dropout = dropout
       

        # 更新输入维度，因为现在只拼接RDKit特征
        input_dim = (concat_dim * head + 20) * 2  # 128 (图特征) + 20 (RDKit特征)
     
        
        self.fc1 = Linear(input_dim, self.pred_dim1)
        self.bn1 = BatchNorm1d(self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.pred_dim2)
        self.bn2 = BatchNorm1d(self.pred_dim2)
        self.fc3 = Linear(self.pred_dim2, self.pred_dim3)
        self.bn3 = BatchNorm1d(self.pred_dim3)
        self.fc4 = Linear(self.pred_dim3, self.out_dim)
        # 残差连接
        self.residual1 = nn.Sequential(
            Linear(input_dim, self.pred_dim1),
            BatchNorm1d(self.pred_dim1)
        )
        
        self.residual2 = nn.Sequential(
            Linear(self.pred_dim1, self.pred_dim2),
            BatchNorm1d(self.pred_dim2)
        )
        
        self.residual3 = nn.Sequential(
            Linear(self.pred_dim2, self.pred_dim3),
            BatchNorm1d(self.pred_dim3)
        )
    
    def forward(self, data):
        x=data
        original_x = x
    
            
        residual = self.residual1(original_x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = x + residual  # 残差连接
        
        # 第二层 + 残差
        residual = self.residual2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = x + residual  # 残差连接
        
        # 第三层 + 残差
        residual = self.residual3(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = x + residual  # 残差连接
        
        # 输出层
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        
        return x
       
        
        #x = self.gcn(xa, edge_index)



class Net1(nn.Module):
    def __init__(self, args):
        super(Net1, self).__init__()
        self.dropout = dropout
        self.conv = args.conv
        
        self.conv1 = GATlayer(n_features,
                              edge_features, 
                              conv_dim1, 
                              conv_dim2, 
                              conv_dim3,
                              concat_dim, 
                              dropout, 
                              )
        self.conv2 = GATlayer(n_features,
                              edge_features, 
                              conv_dim1, 
                              conv_dim2,
                              conv_dim3, 
                              concat_dim, 
                              dropout, 
                              )
        self.fc = FClayer1(concat_dim, 
                              pred_dim1,
                              pred_dim2, 
                              pred_dim3, 
                              out_dim, 
                              dropout)
    
    def forward(self, solute, solvent, device):
        x1 = self.conv1(solute, device)  # Graph Convolution Layer
        x2 = self.conv2(solvent, device)  # Graph Convolution Layer
        x = torch.cat((x1, x2), dim=1)  # Merge
        x = self.fc(x)  # Fully Connected Layer
        return x



import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem
from sklearn.model_selection import train_test_split
import copy
import warnings
warnings.filterwarnings(action='ignore')

# from smiles_to_graph import mol2vec
# from smiles_to_graph import make_regre_mol
# from smiles_to_graph import make_regre_vec
# from regre_exp import experiment
# import regre_model
parser = argparse.ArgumentParser()
args = parser.parse_args('')              

from torch_geometric.nn import GATConv
args.seed=100
args.input_path=r"D:\OLED\absor-final1a.csv"
args.solute_smiles='Chromophore' 
args.solvent_smiles='Solvent' 
args.logS='Absorption max (nm)' 
args.output_path='test_regre.json' 
args.model_path='test_regre_model.pt' 
args.conv='GATConv' # Graph Convolution Type  ('GCNConv', 'SAGEConv', 'ARMAConv', 'GATConv')
args.test_size=0.1
args.random_state=12345
args.batch_size=128
args.epoch=20 #  (Default = 200, Classification)
args.lr=0.001 # Classification
args.step_size=5
args.gamma=0.9
args.exp_name='myExp'
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cpu')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def standardize_features_gloabal(data_list):
    # 处理RDKit特征
    all_features_rdkit = []
    for data in data_list:
        all_features_rdkit.append(data.global_features_rdkit.numpy())
    
    scaler_rdkit = StandardScaler()
    all_features_rdkit = np.vstack(all_features_rdkit)
    scaled_features_rdkit = scaler_rdkit.fit_transform(all_features_rdkit)
    
    start_idx = 0
    for i, data in enumerate(data_list):
        end_idx = start_idx + data.global_features_rdkit.size(0)
        data.global_features_rdkit = torch.tensor(scaled_features_rdkit[start_idx:end_idx], dtype=torch.float)
        start_idx = end_idx
    
    return data_list, (scaler_rdkit)

def standardize_features_with_existing_model_gloabal(data_list, scalers):
    scaler_rdkit = scalers

    # 处理RDKit特征
    all_features_rdkit = []
    for data in data_list:
        all_features_rdkit.append(data.global_features_rdkit.numpy())
    
    all_features_rdkit = np.vstack(all_features_rdkit)
    scaled_features_rdkit = scaler_rdkit.transform(all_features_rdkit)
    
    start_idx = 0
    for i, data in enumerate(data_list):
        end_idx = start_idx + data.global_features_rdkit.size(0)
        data.global_features_rdkit = torch.tensor(scaled_features_rdkit[start_idx:end_idx], dtype=torch.float)
        start_idx = end_idx
    
    return data_list, scalers


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0
    for solute, solvent in train_loader:
        solute = solute.to(device)
        solvent = solvent.to(device)
        optimizer.zero_grad()
        output = model(solute, solvent, device).flatten()
        loss = criterion(output, solute.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, device, loader, y_scaler):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    all_true_orig  = []
    all_pred_orig  = []
    
    with torch.no_grad():
        for solute, solvent in loader:
            solute = solute.to(device)
            solvent = solvent.to(device)
            output = model(solute, solvent, device).flatten()
            loss = criterion(output, solute.y)
            total_loss += loss.item()
            pred_orig = y_scaler.inverse_transform(
                output.cpu().numpy().reshape(-1, 1)
            ).flatten()
            true_orig = solute.y_orig.cpu().numpy()
            
            all_true_orig.extend(true_orig)
            all_pred_orig.extend(pred_orig)
    
    y_true_orig = np.array(all_true_orig)
    y_pred_orig = np.array(all_pred_orig)
    
       
    metrics = {
        'loss': total_loss / len(loader),
        'mae': mean_absolute_error(y_true_orig, y_pred_orig),
        'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
        'r2': r2_score(y_true_orig, y_pred_orig),
        'y_true': y_true_orig,
        'y_pred': y_pred_orig
    }
    return metrics, y_true_orig, y_pred_orig  

print('- Device :', device)
# Train Set, Test Set
import ast
import ast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
df = pd.read_csv(args.input_path)
import joblib


df = pd.concat([df['Tag'], df['Chromophore'], df['Solvent'], df['Absorption max (nm)']], axis=1)
df.columns = ['Tag', 'Chromophore', 'Solvent', 'Absorption max (nm)']

df = df.dropna(axis=0).reset_index(drop=True)

import hashlib
df['hash_value'] = df['Chromophore'].apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100)

train_sizes= [15548]
X_train, X_test = train_test_split(df, test_size=args.test_size, random_state=12345)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
from torch_geometric.data import DataLoader
import copy

mae_results = []
train_losses = []
mae_tr_results =[]
for size in train_sizes:
    print(f"\n--- Training with {size} samples ---")
    X_train_subset = X_train.iloc[:size].reset_index(drop=True)

    
    y_scaler = StandardScaler()

# 使用训练集拟合scaler（只使用训练集）
    y_train_orig = X_train_subset['Absorption max (nm)'].values.reshape(-1, 1)
    y_scaler.fit(y_train_orig)
    X_train_subset['y_scaled'] = y_scaler.transform(y_train_orig).flatten()
    X_test['y_scaled'] = y_scaler.transform(X_test['Absorption max (nm)'].values.reshape(-1, 1)).flatten()
    
    
    
    
    
    print(X_train_subset.shape, X_test.shape)
    print('[Converting to Graph]')

    import joblib
# 转换为图表示
    train_mols1_key, train_mols2_key, train_mols_value = make_regre_mol(X_train_subset)
    test_mols1_key, test_mols2_key, test_mols_value = make_regre_mol(X_test)
    train_X1, train_X2 = make_regre_vec(train_mols1_key, train_mols2_key, X_train_subset['y_scaled'].tolist(), X_train_subset['Absorption max (nm)'].tolist())
    test_X1, test_X2 = make_regre_vec_test(test_mols1_key, test_mols2_key,  X_test['y_scaled'].tolist(), X_test['Absorption max (nm)'].tolist())
    print(train_X1[0])
# 标准化特征
    train_X1, scaler1a = standardize_features_gloabal(train_X1)
    train_X2, scaler2a = standardize_features_gloabal(train_X2)
    test_X1, _ = standardize_features_with_existing_model_gloabal(test_X1,scaler1a)
    test_X2, _ = standardize_features_with_existing_model_gloabal(test_X2,scaler2a)

    joblib.dump(scaler1a, 'scaler1b.pkl')
    joblib.dump(scaler2a, 'scaler2b.pkl')

# 保存y_scaler用于后续逆变换
    joblib.dump(y_scaler, 'y_scaler.pkl')


# 创建数据加载器
    train_data = []
    for i in range(len(train_X1)):
        train_data.append([train_X1[i], train_X2[i]])
    
    test_data = []
    for i in range(len(test_X1)):
        test_data.append([test_X1[i], test_X2[i]])
   

    from torch_geometric.data import DataLoader
 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    
    # 初始化模型
    model = Net1(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # 训练循环
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epoch}")
        print(f"Train Loss: {train_loss:.4f}")
    
    # 测试集评估
    test_metrics, y_true_orig, y_pred_orig = evaluate(model, device, test_loader, y_scaler)
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    
    # 保存结果
    torch.save(model.state_dict(), args.model_path)
    mae_results.append({ 'test_mae': test_metrics['mae']})
    train_metrics, y_true_orig, y_pred_orig = evaluate(model, device, train_loader, y_scaler)
    print(f"Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Train MAE: {train_metrics['mae']:.4f}")
    print(f"Train R²: {train_metrics['r2']:.4f}")

import matplotlib.pyplot as plt
# 8. 绘制学习曲线
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, mae_results, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)
plt.xlabel("Training Set Size", fontsize=14)
plt.ylabel("Test MAE (Absorption max (nm)", fontsize=14)
plt.title("Learning Curve: Test MAE vs Training Set Size (No Validation Set)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(train_sizes)
plt.tight_layout()
plt.show()
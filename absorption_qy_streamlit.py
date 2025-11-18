import streamlit as st
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib
import sys
import os
from rdkit.Chem import rdchem
from rdkit.Chem import Draw
# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从原始文件中提取必要函数，避免触发训练过程
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

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

# 定义模型架构
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import global_add_pool, GATConv

# 吸收波长模型参数
abs_n_features = 38
abs_conv_dim1 = 64
abs_conv_dim2 = 64
abs_conv_dim3 = 64
abs_concat_dim = 32
abs_pred_dim1 = 64
abs_pred_dim2 = 64
abs_pred_dim3 = 64
abs_out_dim = 1
abs_head = 8
abs_edge_features = 6  # 修正为正确的边特征数
abs_dropout = 0.33

# 量子产率模型参数
qy_n_features = 38
qy_conv_dim1 = 64
qy_conv_dim2 = 64
qy_conv_dim3 = 64
qy_concat_dim = 32
qy_pred_dim1 = 64
qy_pred_dim2 = 64
qy_pred_dim3 = 64
qy_out_dim = 1
qy_head = 8
qy_edge_features = 6  # 修正为正确的边特征数
qy_dropout = 0.34

class GATlayer(nn.Module):
    def __init__(self, n_features, edge_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout, head):
        super(GATlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.concat_dim = concat_dim
        self.dropout = dropout
        self.head = head
        
        # 使用 GATConv 层
        self.conv1 = GATConv(self.n_features, self.conv_dim1, self.head, concat=True, edge_dim=edge_features)
        self.bn1 = BatchNorm1d(self.conv_dim1 * self.head)
        self.residual1 = nn.Sequential(
            Linear(n_features, conv_dim1 * head),
            BatchNorm1d(conv_dim1 * head)
        )
     
        self.conv2 = GATConv(self.conv_dim1 * self.head, self.conv_dim2, self.head, concat=True, edge_dim=edge_features)
        self.bn2 = BatchNorm1d(self.conv_dim2 * self.head)
        self.residual2 = nn.Sequential(
            Linear(conv_dim1 * head, conv_dim2 * head),
            BatchNorm1d(conv_dim2 * head))
        
        self.conv3 = GATConv(self.conv_dim2 * self.head, self.conv_dim3, self.head, concat=True, edge_dim=edge_features)
        self.bn3 = BatchNorm1d(self.conv_dim3 * self.head)
        self.residual3 = nn.Sequential(
            Linear(conv_dim2 * head, conv_dim3 * head),
            BatchNorm1d(conv_dim3 * head))
        
        self.conv4 = GATConv(self.conv_dim3 * self.head, self.concat_dim, self.head, concat=True, edge_dim=edge_features)
        self.bn4 = BatchNorm1d(self.concat_dim * self.head)

    def forward(self, data, device):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        global_features_rdkit = data.global_features_rdkit.to(device)
        
        original_x = x
        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = self.bn1(x1)
    
        res1 = self.residual1(original_x)
        x1 = x1 + res1

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x2 = self.bn2(x2)
        res2 = self.residual2(x1)
        x2 = x2 + res2
        
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        x3 = self.bn3(x3)
        res3 = self.residual3(x2)
        x3 = x3 + res3
        
        x = F.relu(self.conv4(x3, edge_index, edge_attr))
        x = self.bn4(x)

        x = global_add_pool(x, data.batch)
        x = torch.cat((x, global_features_rdkit), dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class FClayer1(nn.Module):
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout, head):
        super(FClayer1, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.out_dim = out_dim
        self.dropout = dropout
        self.head = head

        input_dim = (concat_dim * head + 20) * 2 
        
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
        x = data
        original_x = x
            
        residual = self.residual1(original_x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = x + residual

        residual = self.residual2(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = x + residual

        residual = self.residual3(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = x + residual

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        
        return x

class Net1(nn.Module):
    def __init__(self, n_features, edge_features, conv_dim1, conv_dim2, conv_dim3, 
                 concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout, head):
        super(Net1, self).__init__()
        self.dropout = dropout
        
        self.conv1 = GATlayer(n_features, edge_features, conv_dim1, conv_dim2, conv_dim3,
                              concat_dim, dropout, head)
        self.conv2 = GATlayer(n_features, edge_features, conv_dim1, conv_dim2, conv_dim3,
                              concat_dim, dropout, head)
        self.fc = FClayer1(concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout, head)
    
    def forward(self, solute, solvent, device):
        x1 = self.conv1(solute, device)
        x2 = self.conv2(solvent, device)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x
from torch_geometric.data import Data
# 分子转图表示
def mol2vec_abs(mol):
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

def mol2vec_qy(mol):
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

# 标准化函数
def standardize_features_global(data_list, scaler_rdkit):
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
    
    return data_list

# Streamlit界面
st.title(" OLED材料性质预测")

st.markdown("""
本工具可以预测OLED材料的两个关键性质：
1. 吸收波长 (Absorption max - nm)
2. 量子产率 (Quantum Yield)

请输入染料分子和溶剂的SMILES表示来进行预测。
""")

# 用户输入
col1, col2 = st.columns(2)
with col1:
    chromophore_smiles = st.text_input("染料分子 SMILES", "CCOc1ccc(cc1)NC(=O)C")
with col2:
    solvent_smiles = st.text_input("溶剂 SMILES", "CCO")

# 加载模型和标准化器
@st.cache_resource
def load_models_and_scalers():
    # 加载吸收波长模型
    abs_model = Net1(abs_n_features, abs_edge_features, abs_conv_dim1, abs_conv_dim2, abs_conv_dim3,
                     abs_concat_dim, abs_pred_dim1, abs_pred_dim2, abs_pred_dim3, abs_out_dim, 
                     abs_dropout, abs_head)
    abs_model.load_state_dict(torch.load("test_regre_model.pt", map_location=torch.device('cpu')))
    abs_model.eval()
    
    # 加载量子产率模型
    qy_model = Net1(qy_n_features, qy_edge_features, qy_conv_dim1, qy_conv_dim2, qy_conv_dim3,
                    qy_concat_dim, qy_pred_dim1, qy_pred_dim2, qy_pred_dim3, qy_out_dim, 
                    qy_dropout, qy_head)
    qy_model.load_state_dict(torch.load("test_regre_model_qy.pt", map_location=torch.device('cpu')))
    qy_model.eval()
    
    # 加载标准化器
    scaler1b = joblib.load("scaler1b.pkl")
    scaler2b = joblib.load("scaler2b.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    
    scaler1a = joblib.load("scaler1a_qy.pkl")
    scaler2a = joblib.load("scaler2a_qy.pkl")
    
    return abs_model, qy_model, scaler1b, scaler2b, y_scaler, scaler1a, scaler2a

# 预测函数
def predict_properties(chromo_smiles, solvent_smiles):
    try:
        # 创建分子对象
        chromo_mol = Chem.MolFromSmiles(chromo_smiles)
        solvent_mol = Chem.MolFromSmiles(solvent_smiles)
        
        if chromo_mol is None or solvent_mol is None:
            return None, "无法解析SMILES字符串，请检查输入"
        
        # 加载模型和标准化器
        abs_model, qy_model, scaler1b, scaler2b, y_scaler, scaler1a, scaler2a = load_models_and_scalers()
        
        # 提取特征
        chromo_data_abs = mol2vec_abs(chromo_mol)
        solvent_data_abs = mol2vec_abs(solvent_mol)
        
        chromo_data_qy = mol2vec_qy(chromo_mol)
        solvent_data_qy = mol2vec_qy(solvent_mol)
        
        # 标准化
        chromo_data_abs = standardize_features_global([chromo_data_abs], scaler1b)[0]
        solvent_data_abs = standardize_features_global([solvent_data_abs], scaler2b)[0]
        
        chromo_data_qy = standardize_features_global([chromo_data_qy], scaler1a)[0]
        solvent_data_qy = standardize_features_global([solvent_data_qy], scaler2a)[0]
        
        # 进行预测
        with torch.no_grad():
            # 吸收波长预测
            abs_pred = abs_model(chromo_data_abs, solvent_data_abs, torch.device('cpu')).flatten()
            abs_pred_orig = y_scaler.inverse_transform(abs_pred.numpy().reshape(-1, 1)).flatten()[0]
            
            # 量子产率预测
            qy_pred = qy_model(chromo_data_qy, solvent_data_qy, torch.device('cpu')).flatten()
            qy_pred_orig = qy_pred.numpy()[0]
            
        return {
            "absorption": abs_pred_orig,
            "quantum_yield": qy_pred_orig
        }, None
        
    except Exception as e:
        return None, f"预测过程中发生错误: {str(e)}"

# 执行预测
if st.button("预测性质"):
    if chromophore_smiles and solvent_smiles:
        with st.spinner("正在预测..."):
            results, error = predict_properties(chromophore_smiles, solvent_smiles)
            
            if error:
                st.error(error)
            elif results:
                st.success("预测完成！")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="吸收波长 (nm)",
                        value=f"{results['absorption']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="量子产率",
                        value=f"{results['quantum_yield']:.4f}"
                    )
                
                # 显示分子结构
                try:
                    chromo_mol = Chem.MolFromSmiles(chromophore_smiles)
                    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                    
                    st.subheader("分子结构可视化")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("染料分子:")
                        st.image(Draw.MolToImage(chromo_mol), caption="染料分子", width=200)
                    
                    with col2:
                        st.write("溶剂:")
                        st.image(Draw.MolToImage(solvent_mol), caption="溶剂", width=200)
                except Exception as e:
                    st.warning("无法显示分子结构图像")
    else:
        st.warning("请输入染料分子和溶剂的SMILES")

# 示例说明
st.subheader("使用示例")
st.markdown("""
以下是一些常用的SMILES示例：

**染料分子:**
- 苯乙烯基萘衍生物: `C=C(C1=CC=CC=C1)C1=CC=CC2=CC=CC=C12`
- 香豆素衍生物: `C1=CC(=O)OC1=CC1=CC=CC=C1`

**溶剂:**
- 甲苯: `CC1=CC=CC=C1`
- 四氢呋喃: `C1CCOC1`
- 二氯甲烷: `ClCCl`

注意：模型性能最佳的是训练集中出现过的化合物类型。
""")


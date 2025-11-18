import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import numpy as np
import joblib
import argparse
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, global_add_pool

# 设置页面配置
st.set_page_config(page_title="OLED材料性能预测", layout="wide")

# 定义模型架构（从提供的代码中提取）
n_features = 38
conv_dim1 = 64
conv_dim2 = 64
conv_dim3 = 64
concat_dim = 32
pred_dim1 = 64
pred_dim2 = 64
pred_dim3 = 64
out_dim = 1
head = 8
edge_features_abs = 8  # 吸收波长模型的边特征数
edge_features_qy = 6    # 量子产率模型的边特征数
dropout = 0.33

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
        
        self.conv1 = GATConv(self.n_features, self.conv_dim1, self.head, concat=True)
        self.bn1 = BatchNorm1d(self.conv_dim1 * self.head)
        self.residual1 = nn.Sequential(
            Linear(n_features, conv_dim1 * head),
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
        
        # 注意力层（根据保存的模型权重添加）
        self.attention = nn.Sequential(
            nn.Linear(self.concat_dim * self.head * 2, self.concat_dim * self.head),
            nn.ReLU(),
            nn.Linear(self.concat_dim * head, 1),
            nn.Sigmoid())

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
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout):
        super(FClayer1, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.out_dim = out_dim
        self.dropout = dropout

        input_dim = (concat_dim * head + 20) * 2
        
        self.fc1 = Linear(input_dim, self.pred_dim1)
        self.bn1 = BatchNorm1d(self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.pred_dim2)
        self.bn2 = BatchNorm1d(self.pred_dim2)
        self.fc3 = Linear(self.pred_dim2, self.pred_dim3)
        self.bn3 = BatchNorm1d(self.pred_dim3)
        self.fc4 = Linear(self.pred_dim3, self.out_dim)
        
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
    def __init__(self, edge_features):
        super(Net1, self).__init__()
        self.conv1 = GATlayer(n_features, edge_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout)
        self.conv2 = GATlayer(n_features, edge_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout)
        self.fc = FClayer1(concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout)
    
    def forward(self, solute, solvent, device):
        x1 = self.conv1(solute, device)
        x2 = self.conv2(solvent, device)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# 特征处理函数
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=False):
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C','N','O','S','F','Si','P','Cl','Br',
                                                                   'Na','I','B','H', 'Se', 'Sn', 'Te'
                                                                   'Ge']) + 
                           one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5]) + 
                           one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) + 
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

def bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),
                  bond.IsInRing()]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()),
                                                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

from rdkit.Chem import Descriptors, rdMolDescriptors

def calculate_molecular_features(mol):
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

def mol2vec(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom) for atom in atoms]
    global_features_rdkit = calculate_molecular_features(mol)
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]
    
    for bond in bonds:
        edge_attr.append(bond_features(bond))

    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    data.global_features_rdkit = torch.tensor(global_features_rdkit, dtype=torch.float)
    return data

def standardize_features_with_existing_model_global(data, scaler):
    data.global_features_rdkit = torch.tensor(
        scaler.transform(data.global_features_rdkit.numpy().reshape(1, -1)), 
        dtype=torch.float
    )
    return data

# 加载模型和标准化器
@st.cache_resource
def load_models():
    device = torch.device('cpu')
    
    # 加载吸收波长预测模型 (使用8个边特征)
    abs_model = Net1(edge_features_abs)
    abs_model.load_state_dict(torch.load("test_regre_model.pt", map_location=device))
    abs_model.eval()
    
    # 加载量子产率预测模型 (使用6个边特征)
    qy_model = Net1(edge_features_qy)
    qy_model.load_state_dict(torch.load("test_regre_model_qy.pt", map_location=device))
    qy_model.eval()
    
    # 加载标准化器
    scaler1b = joblib.load("scaler1b.pkl")
    scaler2b = joblib.load("scaler2b.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    scaler1a_qy = joblib.load("scaler1a_qy.pkl")
    scaler2a_qy = joblib.load("scaler2a_qy.pkl")
    
    return abs_model, qy_model, scaler1b, scaler2b, y_scaler, scaler1a_qy, scaler2a_qy

# 主应用程序
def main():
    st.title("OLED材料性能预测")
    st.markdown("""
    该应用程序可以预测OLED材料的两个关键性能指标：
    - **吸收波长** (Absorption max in nm)
    - **量子产率** (Quantum Yield)
    
    请输入分子的SMILES表示和溶剂SMILES来获取预测结果。
    """)
    
    # 侧边栏输入
    st.sidebar.header("输入参数")
    
    chromophore_smiles = st.sidebar.text_input(
        "发光团 SMILES",
        placeholder="例如: C1=CC=C(C=C1)C2=CC=CC=C2"
    )
    
    solvent_smiles = st.sidebar.text_input(
        "溶剂 SMILES",
        placeholder="例如: CCO (乙醇)"
    )
    
    # 加载模型
    try:
        abs_model, qy_model, scaler1b, scaler2b, y_scaler, scaler1a_qy, scaler2a_qy = load_models()
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return
    
    # 验证SMILES
    try:
        chromophore_mol = Chem.MolFromSmiles(chromophore_smiles)
        solvent_mol = Chem.MolFromSmiles(solvent_smiles)
        
        if chromophore_mol is None:
            st.sidebar.error("发光团 SMILES 无效")
            return
            
        if solvent_mol is None:
            st.sidebar.error("溶剂 SMILES 无效")
            return
    except Exception as e:
        st.sidebar.error(f"SMILES解析错误: {str(e)}")
        return
    

    
    # 预测按钮
    if st.button("预测性能", type="primary"):
        with st.spinner("正在预测..."):
            try:
                # 转换为图表示
                chromophore_data = mol2vec(chromophore_mol)
                solvent_data = mol2vec(solvent_mol)
                
                # 标准化特征（吸收波长模型）
                chromophore_data_abs = standardize_features_with_existing_model_global(
                    chromophore_data, scaler1b)
                solvent_data_abs = standardize_features_with_existing_model_global(
                    solvent_data, scaler2b)
                
                # 标准化特征（量子产率模型）
                chromophore_data_qy = standardize_features_with_existing_model_global(
                    chromophore_data, scaler1a_qy)
                solvent_data_qy = standardize_features_with_existing_model_global(
                    solvent_data, scaler2a_qy)
                
                device = torch.device('cpu')
                
                # 吸收波长预测
                with torch.no_grad():
                    abs_pred = abs_model(chromophore_data_abs, solvent_data_abs, device)
                    abs_pred_orig = y_scaler.inverse_transform(
                        abs_pred.cpu().numpy().reshape(-1, 1)
                    ).flatten()[0]
                
                # 量子产率预测
                with torch.no_grad():
                    qy_pred = qy_model(chromophore_data_qy, solvent_data_qy, device)
                    qy_pred_value = qy_pred.cpu().numpy()[0][0]
                
                # 显示结果
                st.subheader("预测结果")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="吸收波长 (nm)",
                        value=f"{abs_pred_orig:.2f}",
                        delta="预测值"
                    )
                
                with col2:
                    st.metric(
                        label="量子产率",
                        value=f"{qy_pred_value:.4f}",
                        delta="预测值"
                    )
                
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}")
                st.error("请检查输入的SMILES是否正确")
    # 显示分子结构
    st.subheader("分子结构")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**发光团结构**")
        try:
            chromophore_img = Draw.MolToImage(chromophore_mol)
            st.image(chromophore_img, caption="发光团分子",  use_container_width=True)
        except Exception as e:
            st.error(f"无法绘制发光团结构: {str(e)}")
    
    with col2:
        st.markdown("**溶剂结构**")
        try:
            solvent_img = Draw.MolToImage(solvent_mol)
            st.image(solvent_img, caption="溶剂分子",  use_container_width=True)
        except Exception as e:
            st.error(f"无法绘制溶剂结构: {str(e)}")
if __name__ == "__main__":
    main()

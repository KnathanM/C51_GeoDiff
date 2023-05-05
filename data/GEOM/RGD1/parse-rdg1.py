import h5py
import numpy as np
from rdkit.Geometry import Point3D
from rdkit import Chem
import torch
from torch_geometric.data import Data, Dataset
from torch_scatter import scatter
import copy
from rdkit.Chem.rdchem import BondType as BT
import pickle
import random



class ConformationDataset(Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.atom_types = self._atom_types()
        self.edge_types = self._edge_types()

    def __getitem__(self, idx):
        data = self.data[idx].clone()
        if self.transform is not None:
            data = self.transform(data)        
        return data

    def __len__(self):
        return len(self.data)

    def _atom_types(self):
        """All atom types."""
        atom_types = set()
        for graph in self.data:
            atom_types.update(graph.atom_type.tolist())
        return sorted(atom_types)

    def _edge_types(self):
        """All edge types."""
        edge_types = set()
        for graph in self.data:
            edge_types.update(graph.edge_type.tolist())
        return sorted(edge_types)


class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data

# This will parse the rdkit mol object and get it ready for GeoDiff
def rdmol_to_data(mol, Rsmiles, Psmiles, R_G, P_G):
    BOND_TYPES = {t: i for i, t in enumerate(BT.names.values())}
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()
    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)
    reac = torch.tensor(R_G, dtype=torch.float32)
    prod = torch.tensor(P_G, dtype=torch.float32)

    atomic_number = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)
    if edge_index.size(1) == 0: # only alpha carbon
        return None
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    name = Rsmiles
    prod_name = Psmiles
    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, is_alpha=None,
                rdmol=copy.deepcopy(mol), name=name, prod_name=prod_name, is_sidechain=None, atom2res=None, atom2alpha_index=None, RG=reac, PG=prod)
    return data

hf = h5py.File('RGD1_CHNO.h5', 'r')

all_data = []
failure = 0

with open('Rinds.csv') as f:
    rxn_to_include = [line.strip() for line in f]

for Rind,Rxn in hf.items():

    if Rind not in rxn_to_include:
        continue

    print(Rind)
    # parse smiles
    Rsmiles,  Psmiles = str(np.array(Rxn.get('Rsmiles'))),   str(np.array(Rxn.get('Psmiles')))
    
    # parse geometries
    TS_G = np.array(Rxn.get('TSG'))    
    R_G = np.array(Rxn.get('RG'))
    P_G = np.array(Rxn.get('PG'))

    # Make rdkit mol, assign positions, and create data object
    mol_init = Chem.MolFromSmiles(Rsmiles)
    mol = Chem.AddHs(mol_init)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x,y,z = TS_G[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    mol.AddConformer(conf)
    data = rdmol_to_data(mol, Rsmiles, Psmiles, R_G, P_G)
    if data is not None:
        all_data.append(data)
    else:
        failure = failure + 1

# Write to pkl
with open('all_rgd_data_b1f1.pkl', "wb") as fout:
    pickle.dump(all_data, fout)

print(str(failure) + " failures")
print("Data saved!")

# Make train, validation, and test set
train_size = 1400
val_size = 300
test_size = 421
with open("all_rgd_data_b1f1.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

selected_data = random.sample(loaded_data, train_size + val_size + test_size)
train_data = selected_data[0:train_size]
val_data = selected_data[train_size:train_size+val_size]
test_data = selected_data[train_size+val_size:]
print(len(train_data))
print(len(val_data))
print(len(test_data))

with open("train_data_1400.pkl", "wb") as fout:
    pickle.dump(train_data, fout)
with open("val_data_300.pkl", "wb") as fout:
    pickle.dump(val_data, fout)
with open("test_data_421.pkl", 'wb') as fout:
    pickle.dump(test_data, fout)

# Check if created data correctly
with open("train_data_1400.pkl", "rb") as f:
    train_data_load = pickle.load(f)
    print(train_data_load[0])

# Sanity check to make sure it will work in GeoDiff
transforms = CountNodesPerGraph()
train_set = ConformationDataset("train_data_1400.pkl", transform=transforms)

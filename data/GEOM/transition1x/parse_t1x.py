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
from xyz2mol_edited import *



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
def rdmol_to_data(mol, R_G, P_G):
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

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type, is_alpha=None,
                rdmol=copy.deepcopy(mol), is_sidechain=None, atom2res=None, atom2alpha_index=None, RG=reac, PG=prod)
    return data

hf = h5py.File('transition1x.h5', 'r')['data']

all_data = []
failure = 0
z_to_l = [' ','h','he','li','be','b','c','n','o','f']
counter = 0
for formula, grp in hf.items():
    for rxn, subgrp in grp.items():
        counter += 1
        # parse geometries
        TS_G = np.array(subgrp["transition_state"].get('positions')).squeeze()
        R_G = np.array(subgrp["reactant"].get('positions')).squeeze()
        P_G = np.array(subgrp["product"].get('positions')).squeeze()

        z = np.array(subgrp.get("atomic_numbers"))
        
        Rsmiles = xyz2mol_nathan(z.tolist(),R_G.tolist())
        print(counter,"SMILES:",Rsmiles)
        if Rsmiles == "Failed":
            print("No SMILES")
            failure = failure + 1
            continue
        
        # Make rdkit mol, assign positions, and create data object
        mol_init = Chem.MolFromSmiles(Rsmiles)
        mol = Chem.AddHs(mol_init)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x,y,z = TS_G[i]
            conf.SetAtomPosition(i,Point3D(x,y,z))
        mol.AddConformer(conf)
        data = rdmol_to_data(mol, R_G, P_G)
        if data is not None:
            all_data.append(data)
        else:
            failure = failure + 1

# Write to pkl
with open('all_t1x_data.pkl', "wb") as fout:
    pickle.dump(all_data, fout)

print(str(failure) + " failures")
print("Data saved!")

# Make train, validation, and test set
train_size = 8000
val_size = 1000
test_size = 1073 - failure
with open("all_t1x_data.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

selected_data = random.sample(loaded_data, train_size + val_size + test_size)
train_data = selected_data[0:train_size]
val_data = selected_data[train_size:train_size+val_size]
test_data = selected_data[train_size+val_size:]
print(len(train_data))
print(len(val_data))
print(len(test_data))

with open("train_data_8000.pkl", "wb") as fout:
    pickle.dump(train_data, fout)
with open("val_data_1000.pkl", "wb") as fout:
    pickle.dump(val_data, fout)
with open("test_data_1073.pkl", 'wb') as fout:
    pickle.dump(test_data, fout)

# Check if created data correctly
with open("train_data_8000.pkl", "rb") as f:
    train_data_load = pickle.load(f)
    print(train_data_load[0])

# Sanity check to make sure it will work in GeoDiff
transforms = CountNodesPerGraph()
train_set = ConformationDataset("train_data_8000.pkl", transform=transforms)

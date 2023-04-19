import pickle
from copy import deepcopy
import rdkit
from rdkit.Chem import rdMolAlign as MA
import statistics

def set_rdmol_positions(rdkit_mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    set_rdmol_positions_(mol, pos)
    return mol

def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol

with open('samples_all.pkl', 'rb') as f:
    dataset = pickle.load(f)

all_rmsd = []
for data in dataset:
    mol = data['rdmol']
    pos_ref = data['pos_ref']
    pos_gen_all = data['pos_gen']
    pos_gen = pos_gen_all[0:len(pos_ref),:]
    pos_gen2 = pos_gen_all[len(pos_ref):,:]
    mol_ref = set_rdmol_positions(mol, pos_ref)
    mol_gen = set_rdmol_positions(mol, pos_gen)
    mol_gen2 = set_rdmol_positions(mol, pos_gen2)
    rmsd = MA.GetBestRMS(mol_gen, mol_ref)
    rmsd2 = MA.GetBestRMS(mol_gen2, mol_ref)
    all_rmsd.append(min(rmsd2,rmsd))

print(statistics.mean(all_rmsd))
print(statistics.stdev(all_rmsd))
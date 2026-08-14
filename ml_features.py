import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


FP_RADIUS = 2
FP_SIZE = 2048

BASE_DESCRIPTORS = [
    "MolWt",
    "HeavyAtomCount",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "NHOHCount",
    "NOCount",
    "NumHeteroatoms",
    "NumValenceElectrons",
    "BalabanJ",
    "BertzCT",
    "Chi0v",
    "Chi1v",
    "Kappa1",
    "Kappa2",
]

LOGS_EXTRA_DESCRIPTORS = ["MolLogP"]

DESC_FUNCS = dict(Descriptors._descList)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS,
    fpSize=FP_SIZE,
)


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold

    return Chem.MolToSmiles(mol, canonical=True)


def descriptor_names(task):
    names = list(BASE_DESCRIPTORS)

    if task == "logs":
        names.extend(LOGS_EXTRA_DESCRIPTORS)

    return names


def featurize_mol(mol, task):
    values = {}
    names = descriptor_names(task)

    for name in names:
        func = DESC_FUNCS.get(name)
        if func is None:
            values[f"desc_{name}"] = 0.0
            continue

        try:
            value = float(func(mol))
            if not np.isfinite(value):
                value = 0.0
        except Exception:
            value = 0.0

        values[f"desc_{name}"] = value

    fp = MORGAN.GetFingerprintAsNumPy(mol)

    for i, bit in enumerate(fp):
        values[f"fp_{i}"] = int(bit)

    return values


def featurize_smiles(smiles, task):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return featurize_mol(mol, task)


def make_feature_frame(smiles_list, task):
    rows = []
    valid_smiles = []
    valid_indices = []

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol, canonical=True)
        rows.append(featurize_mol(mol, task))
        valid_smiles.append(canonical)
        valid_indices.append(idx)

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, valid_smiles, valid_indices


def make_single_feature_frame(smiles, task, feature_names):
    row = featurize_smiles(smiles, task)
    if row is None:
        return None

    return pd.DataFrame([row]).reindex(
        columns=feature_names,
        fill_value=0.0,
    )

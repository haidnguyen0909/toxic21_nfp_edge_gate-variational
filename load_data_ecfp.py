import rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy
import chainer.datasets as D
import chainer.datasets.tuple_dataset as Tuple


filename_train = 'tox21_10k_data_all.sdf'
filename_val = 'tox21_10k_challenge_test.sdf'

label_names = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
               'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
               'SR-HSE', 'SR-MMP', 'SR-p53']

MAX_NUMBER_ATOM = 140

def construct_edge_matrix(mol):
    if mol is None:
        return None
    N = mol.GetNumAtoms()
    size = MAX_NUMBER_ATOM
    adjs = numpy.zeros((4, size, size), dtype=numpy.float32)
    for i in range(N):
        for j in range(N):
            bond = mol.GetBondBetweenAtoms(i, j)  # type: Chem.Bond
            if bond is not None:
                bondType = str(bond.GetBondType())
                if bondType == 'SINGLE':
                    adjs[0, i, j] = 1.0
                elif bondType == 'DOUBLE':
                    adjs[1, i, j] = 1.0
                elif bondType == 'TRIPLE':
                    adjs[2, i, j] = 1.0
                elif bondType == 'AROMATIC':
                    adjs[3, i, j] = 1.0
                else:
                    print("[ERROR] Unknown bond type", bondType)
                    assert False  # Should not come here
    return adjs

def preprocessor(mol_supplier, label_names):
    descriptors = []
    labels = []
    count = 0
    for mol in mol_supplier:
        if mol is None:
            continue

        label = []
        for task in label_names:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)

        #adj = rdmolops.GetAdjacencyMatrix(mol)
        adj = construct_edge_matrix(mol)
        atom_list = [a.GetSymbol() for a in mol.GetAtoms()]
        labels.append(label)
        descriptors.append((adj, atom_list))
        if count == 10000:
            break
        count += 1

    labels = numpy.array(labels, dtype=numpy.int32)
    return descriptors, labels

def getAtom2id(train, val):
    max_atom = 0
    for data in [train, val]:
        for d in data:
            adj = d[0][0]

            atom_list = d[0][1]
            max_atom = max(max_atom, len(atom_list))
    assert max_atom <= MAX_NUMBER_ATOM

    # Construct atom2id dictionary
    atom2id = {'empty': 0}
    atoms = [d[0][1] for d in train] + [d[0][1] for d in val]
    atoms = sum(atoms, [])
    for a in atoms:
        if a not in atom2id:
            atom2id[a] = len(atom2id)

    train = convert_dataset(train, atom2id)
    val = convert_dataset(val, atom2id)
    return train, val, atom2id




def get_tox21():
    molSupplier_train = Chem.SDMolSupplier(filename_train)
    molSupplier_val = Chem.SDMolSupplier(filename_val)
    descriptors_train, label_train = preprocessor(molSupplier_train, label_names)
    descriptors_val, label_val = preprocessor(molSupplier_val, label_names)
    return Tuple.TupleDataset(descriptors_train, label_train), Tuple.TupleDataset(descriptors_val, label_val)



def convert_dataset(dataset, atom2id):
    ret = []
    for d in dataset:
        (adj, atom_list), label = d

        # 0 padding for adj matrix
        #s0, s1 = adj.shape
        #adj = adj + numpy.eye(s0)
        #adj_array = numpy.zeros((MAX_NUMBER_ATOM, MAX_NUMBER_ATOM),
        #                        dtype=numpy.float32)

        #adj_array[:s0, :s1] = adj.astype(numpy.float32)
        # print('adj_array', adj_array)

        # 0 padding for atom_list
        atom_list = [atom2id[a] for a in atom_list]
        n_atom = len(atom_list)
        atom_array = numpy.zeros((MAX_NUMBER_ATOM,), dtype=numpy.int32)
        atom_array[:n_atom] = numpy.array(atom_list)

        ret.append((adj, atom_array, label))
    return ret

def load_one_task(task, filename, atom2id):
    molSupplier = Chem.SDMolSupplier(filename)
    descriptors = []
    labels = []
    count = 0
    for mol in molSupplier:
        if mol is None:
            continue

        label = []
        for _ in label_names:
            if mol.HasProp(task):
                label.append(int(mol.GetProp(task)))
            else:
                label.append(-1)

        # adj = rdmolops.GetAdjacencyMatrix(mol)
        adj = construct_edge_matrix(mol)
        atom_list = [a.GetSymbol() for a in mol.GetAtoms()]
        labels.append(label)
        descriptors.append((adj, atom_list))

    labels = numpy.array(labels, dtype=numpy.int32)
    dataset = Tuple.TupleDataset(descriptors, labels)
    dataset = convert_dataset(dataset, atom2id)
    return dataset




def make_dataset():
    train, test = get_tox21()
    train, test, atom2id = getAtom2id(train, test)
    train, val = D.split_dataset(train, int(0.9 * len(train)))



    print("size of train set:", len(train))

    print("size of val set:", len(val))
    print('size of test set:', len(test))

    return train, val, test, atom2id

make_dataset()
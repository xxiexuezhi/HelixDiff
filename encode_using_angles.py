import numpy as np
import scipy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import logging
import warnings
from tqdm.contrib.concurrent import process_map
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from torch.utils.data._utils.collate import default_collate

non_standard_to_standard = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'ASX':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA', # Added ASX => ASP
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYL':'LYS', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS', 'SEC':'CYS', # Added pyrrolysine and selenocysteine
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

three_to_one_letter = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M', 'UNK': 'X'}

one_to_three_letter = {v:k for k,v in three_to_one_letter.items()}

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}


import prody as pr
import sidechainnet as scn
from sidechainnet.utils.download import get_resolution_from_pdbid
import torch
def process_pdb(filename, pdbid="", include_resolution=False):
    """Return a dictionary containing SidechainNet-relevant data for a given PDB file.

    Args:
        filename (str): Path to existing PDB file.
        pdbid (str): 4-letter string representing the PDB Identifier.
        include_resolution (bool, default=False): If True, query the PDB for the protein
            structure resolution based off of the given pdb_id.

    Returns:
        scndata (dict): A dictionary holding the parsed data attributes of the protein
        structure. Below is a description of the keys:

            The key 'seq' is a 1-letter amino acid sequence.
            The key 'coords' is a (L x NUM_COORDS_PER_RES) x 3 numpy array.
            The key 'angs' is a L x NUM_ANGLES numpy array.
            The key 'is_nonstd' is a L x 1 numpy array with binary values. 1 represents
                that the amino acid at that position was a non-standard amino acid that
                has been modified by SidechainNet into its standard form.
            The key 'unmodified_seq' refers to the original amino acid sequence
                of the protein structure. Some non-standard amino acids are converted into
                their standard form by SidechainNet before measurement. In this case, the
                unmodified_seq variable will contain the original (3-letter code) seq.
            The key 'resolution' is the resolution of the structure as listed on the PDB.
    """
    # First, use Prody to parse the PDB file
    chain = pr.parsePDB(filename)
    # Next, use SidechainNet to make the relevant measurements given the Prody chain obj
    (dihedrals_np, coords_np, observed_sequence, unmodified_sequence,
     is_nonstd) = scn.utils.measure.get_seq_coords_and_angles(chain, replace_nonstd=True)
    scndata = {
        "coords": coords_np,
        "angs": dihedrals_np,
        "seq": observed_sequence,
        "unmodified_seq": unmodified_sequence,
        "is_nonstd": is_nonstd
    }
    # If requested, look up the resolution of the given PDB ID
    if include_resolution:
        assert pdbid, "You must provide a PDB ID to look up the resolution."
        scndata['resolution'] = get_resolution_from_pdbid(pdbid)
    return scndata



import numpy as np
#import perfplot
import math
import numpy as np

def encode_angle_sin_cos(angle):
    return np.sin(angle),np.cos(angle)

def f(x):
    # return math.sqrt(x)
    return encode_angle_sin_cos(x)


vf = np.vectorize(f)


def array_for(x):
    return np.array([f(xi) for xi in x])


def encode_ang_using_pdb(pdb_name):
    a = process_pdb(pdb_name)
    a_angle = a["angs"]
    a_seq = a["seq"]
    a_encoded = array_for(a_angle).transpose([0,2,1])
    a_encoded_reshape = a_encoded.reshape(128,24)
    return a_seq,a_encoded_reshape

def f2(arr12_2): #arr2 size is (12,2) return 12
    # return math.sqrt(x)
    return np.array([np.arctan2(arr2[0],arr2[1]) for arr2 in arr12_2])


#vf = np.vectorize(f)


def array_for2(x):
    return np.array([f2(xi) for xi in x])

# taking the input array size as 128, 24. change it into angles for sidechainnet which is 128, 12
def decode_to_angles(arr):
    arr_reshape = arr.reshape(128,12,2)
    return array_for2(arr_reshape)

    #arr_reshape_permute = arr_reshape_permute([0,2,1])

def one_hot_encode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(alphabet,range(len(alphabet))))
    seq_idx = [mapping[s] for s in seq]
    return np.eye(len(alphabet))[seq_idx]

def one_hot_decode(seq):
    alphabet = "ARNDCEQGHILKMFPSTWYV"
    mapping = dict(zip(range(len(alphabet)),alphabet))
    seq_idx = [np.argmax(i) for i in seq]
    return ''.join([mapping[i] for i in seq_idx])


def get_angles_to_relace_6d(path):
    a = process_pdb(str(path))
    coords = a["coords"]
            #print()
            #return None
    a_angle = a["angs"]
    a_seq = a["seq"]
    if len(a_seq)!= 128:
        return None
    seq = a_seq
    a_encoded = array_for(a_angle).transpose([0,2,1])
    a_encoded_reshape = a_encoded.reshape(128,24)
    a_encoded_reshape_pad = torch.zeros([2,128,128])
    mid_ptr = 64

    mid_point = mid_ptr
    a_encoded_reshape_pad[0,:,64-12:64+12] = torch.from_numpy(a_encoded_reshape)
    a_encoded_reshape_pad[1,:,mid_point-10:mid_point+10] = torch.from_numpy(one_hot_encode(seq))
    x = 
    if torch.count_nonzero(x) == 0 or torch.isnan(x).any():
        print(path)
        return None
    return a_encoded_reshape_pad





class ProteinDataset(Dataset):
    def __init__(self, dataset_path, min_res_num=40, max_res_num=256, ss_constraints=True):
        super().__init__()
        # Ignore biotite warnings
        warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

        self.min_res_num = min_res_num
        self.max_res_num = max_res_num
        self.ss_constraints = ss_constraints

        # Load PDB files into dataset
        paths = list(Path(dataset_path).iterdir())
        structures = self.parse_pdb(paths)

        # Remove None from self.structures
        self.structures = [self.to_tensor(i) for i in structures if i is not None]

    def parse_pdb(self, paths):
        logging.info(f"Processing dataset of length {len(paths)}...")
        data = list(process_map(self.get_features, paths, chunksize=10))
        return data

    def get_coarse_constraints(self, model, cb, dist_threshold=7, dmax=20, block_dropout=0.1):
        # Used for splitting block secondary structures
        def consecutive(data, stepsize=1):
            return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

        dist_threshold_norm = (dist_threshold / dmax * 2) - 1

        psea_to_index = {"a": 1, "b": 2, "c": 3}
        chain_id = struc.get_chains(model)[0]
        s = [psea_to_index[i] for i in struc.annotate_sse(model, chain_id)]
        if len(s) != cb.shape[0]: return None, None  # Shape mismatch from PSEA: TODO: Find issue
        # annotate_sse is based on CA coordinates, so the shape is wrong if a CA coordinate is missing
        # Fix by inserting 0 at indices where CA coordinates are missing
        # ca_mask_index = (1-ca_atom_mask).nonzero()[0]
        # [s.insert(i,0) for i in ca_mask_index]
        s = np.array(s)

        helix_mask = (s == 1)
        beta_mask = (s == 2)

        # Block adjacencies
        helix_indices = helix_mask.nonzero()[0]
        beta_indices = beta_mask.nonzero()[0]

        helix_indices_split = [i for i in consecutive(helix_indices) if len(i) >= 4]
        beta_indices_split = [i for i in consecutive(beta_indices) if len(i) >= 4]

        helix_mask_pair = np.zeros(cb.shape)
        for i in helix_indices_split:
            start, end = i[0], i[-1]
            helix_mask_pair[start:end, start:end] = 1

        beta_mask_pair = np.zeros(cb.shape)
        for i1 in beta_indices_split:
            for i2 in beta_indices_split:
                start1, end1 = i1[0], i1[-1]
                start2, end2 = i2[0], i2[-1]
                beta_mask_pair[start1:end1, start2:end2] = 1

        helix_beta_indices = helix_indices_split + beta_indices_split

        block_adj_mask = np.zeros(cb.shape)
        for idx1, block1 in enumerate(helix_beta_indices):
            for idx2, block2 in enumerate(helix_beta_indices):
                if idx1 == idx2: continue
                b1_start, b1_end = block1[0], block1[-1]
                b2_start, b2_end = block2[0], block2[-1]
                dist = cb[b1_start:b1_end, b2_start:b2_end].min()
                if dist < dist_threshold_norm:
                    block_adj_mask[b1_start:b1_end, b2_start:b2_end] = 1
        constraints = np.stack([helix_mask_pair, beta_mask_pair, block_adj_mask], axis=-1)

        # Convert to string for dataloader
        helix_beta_str = ','.join([f"{i[0]}:{i[-1]}" for i in helix_beta_indices])
        return constraints, helix_beta_str

    def get_features(self, path):
        with open(path, "r") as f:
            structure = PDBFile.read(f)

        if structure.get_model_count() > 1: return None
        struct = structure.get_structure()
        if struc.get_chain_count(struct) > 1: return None
        _, aa = struc.get_residues(struct)

        # Replace nonstandard amino acids
        for idx,a in enumerate(aa):
            if a not in three_to_one_letter.keys():
                aa[idx] = non_standard_to_standard.get(a, "UNK")

        one_letter_aa = [three_to_one_letter[i] for i in aa]
        aa_str = ''.join(one_letter_aa)
        aa = [letter_to_num[i] for i in one_letter_aa]
        nres = len(aa)

        if nres > self.max_res_num or nres < self.min_res_num: return None

        mask = np.ones(nres)
        atom_mask = np.ones((nres, 3))

        bb_coords = []
        for res_idx, res in enumerate(struc.residue_iter(struct)):
            # Find backbone + CB atoms
            atom_types = res.get_annotation("atom_name")
            all_coords = res.coord[0]
            crd = []
            for atom_idx, a in enumerate(["N", "CA", "C"]):
                idx = np.where(atom_types == a)[0]
                if idx.size == 0:
                    atom_mask[res_idx, atom_idx] = 0
                    # Rolling mask i-1 and i+1 since all 3 atoms are used for CB reconstruction
                    mask[res_idx] = 0
                    if res_idx != 0:
                        mask[res_idx-1] = 0
                    if res_idx != nres-1:
                        mask[res_idx+1] = 0
                    crd.append([0, 0, 0])
                else:
                    crd.append(all_coords[idx[0]])
            bb_coords.append(crd)
        bb_coords = np.array(bb_coords)

        coords_6d = get_coords6d(bb_coords, dmax=20.0, normalize=True)
        coords_6d = np.nan_to_num(coords_6d)
        padding = np.ones((nres,nres)).reshape(nres,nres,1)

        if self.ss_constraints:
            block_adj, helix_beta_str = self.get_coarse_constraints(struct[0], coords_6d[:, :, 0], dist_threshold=5)
            if block_adj is None: return None
            coords_6d = np.concatenate([coords_6d,block_adj,padding],axis=-1)
        else:
            coords_6d = np.concatenate([coords_6d, padding], axis=-1)
            helix_beta_str = []
        mask_pair = mask.reshape(1,-1) * mask.reshape(-1, 1) # N, N

        coords_6d = coords_6d * mask_pair.reshape(nres,nres,1) # N, N, C
        coords_6d = coords_6d.transpose(2,0,1) # C, N, N
        pad_angles_tensor = get_angles_to_relace_6d(str(path))
        if pad_angles_tensor is None: return None
        return {
            "id": path.stem,
            "coords": bb_coords,
            "coords_6d": pad_angles_tensor,
            "aa": aa,
            "aa_str": aa_str,
            "mask_pair": mask_pair,
            "ss_indices": helix_beta_str # Used for block dropout
        }

    def to_tensor(self, d):
        feat_dtypes = {
            "id": None,
            "coords": torch.float32,
            "coords_6d": torch.float32,
            "aa": torch.long,
            "aa_str": None,
            "mask_pair": torch.bool,
            "ss_indices": None
        }

        for k,v in d.items():
            if feat_dtypes[k] is not None:
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

        return d

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx]


##### Functions below adapted from trRosetta https://github.com/RosettaCommons/trRosetta2/blob/main/trRosetta/coords6d.py
# calculate dihedral angles defined by 4 sets of points
def get_dihedrals(a, b, c, d):
    
    # Ignore divide by zero errors
    np.seterr(divide='ignore', invalid='ignore')
    
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c
    
    b1 /= np.linalg.norm(b1, axis=-1)[:,None]
    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

# calculate planar angles defined by 3 sets of points
def get_angles(a, b, c):

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)

# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, dmax=20.0, normalize=True):

    nres = xyz.shape[0]

    # three anchor atoms
    N  = xyz[:,0]
    Ca = xyz[:,1]
    C  = xyz[:,2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres), dmax).astype(float)
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    
    # Normalize all features to [-1,1]
    if normalize:
        # [4A, 20A]
        dist6d = (dist6d / dmax*2) - 1
        # [-pi, pi]
        omega6d = omega6d / math.pi
        # [-pi, pi]
        theta6d = theta6d / math.pi
        # [0, pi]
        phi6d = (phi6d / math.pi*2) - 1

    coords_6d = np.stack([dist6d,omega6d,theta6d,phi6d],axis=-1)

    return coords_6d

class PaddingCollate(object):

    def __init__(self, max_len=None):
        super().__init__()
        self.max_len = max_len

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x

            # Pairwise embeddings TODO: not very elegant
            if len(x.shape) >= 2 and x.shape[-1] != 3 and x.shape[-1] == x.shape[-2]:
                x = F.pad(x, (0,n-x.shape[-1],0,n-x.shape[-2]), value=value)
                return x

            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, str):
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_value(k):
        if k in ["aa_str"]:
            return "_"
        elif k == "aa":
            return 21 # masking value
        elif k in ["id","ss_indices"]:
            return ''
        else:
            return 0

    def __call__(self, data_list):
        max_length = self.max_len if self.max_len else max([len(data["aa"]) for data in data_list])
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_value(k)) for k,v in data.items()
            }
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)



ds = ProteinDataset(
        "",
        max_res_num = 128
    )

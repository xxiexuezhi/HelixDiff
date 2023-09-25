# import matplotlib.pyplot as plt
import subprocess
import tempfile

import numpy as np
import torch
import random
from pathlib import Path
from dataset import ProteinDataset, PaddingCollate
from score_sde_pytorch.utils import recursive_to
from biotite.structure.io import load_structure, save_structure
import biotite.structure as struc
import shutil

import pickle
def random_mask_batch(batch, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask_min = config.model.inpainting.mask_min_len
    mask_max = config.model.inpainting.mask_max_len

    random_mask_prob = config.model.inpainting.random_mask_prob
    contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob

    lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    prob = random.random()
    if prob < random_mask_prob:
        # Random masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            rand_indices = torch.randperm(l)[:rand]

            m = torch.zeros(N)
            m[rand_indices] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    elif prob > 1-contiguous_mask_prob:
        # Contiguous masking
        mask = []
        for l in lengths:
            rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
            index = torch.randint(0, (l - rand).int(), (1,))[0]

            m = torch.zeros(N)
            m[index:index + rand] = 1

            mask.append(m)
        mask = torch.stack(mask, dim=0)
    else:
        mask = torch.ones(B, N) # No masking

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch

def selected_mask_batch(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask = torch.zeros(B, N)

    res_mask = mask_info.split(",")
    for r in res_mask:
        if ":" in r:
            start_idx, end_idx = r.split(":")
            mask[:, int(start_idx):int(end_idx)+1] = 1
        else:
            mask[:,int(r)] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch

# update the mask_info to take it as a list of str num.
def selected_mask_batch2(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    mask = torch.zeros(B, N)
    ss_indices = mask_info
    #res_mask = mask_info.split(",")
    for idx in range(len(mask_info)):
        if mask_info[idx] == '': continue # no secondary structure annotation found
        ss_idx = mask_info[idx].split(",")
        indices_for_dropout = [b for b in ss_idx]
        #for i in indices_for_dropout:
            #start, end = [int(x) for x in i.split(":")]
        for r in indices_for_dropout:
            if ":" in r:
                start_idx, end_idx = r.split(":")
                mask[idx, int(start_idx):int(end_idx)] = 1
            else:
                mask[idx,int(r)] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

    return batch

mid_ptr = 64

# updated on May 25 2023 for helixsgm-inpainting model

def selected_mask_batch3(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape

    mask = torch.ones(B,N,N) # one means all regerentated. 
    #seq_mask = torch.zeros(B,N,N).bool()

    ss_indices = mask_info # I updated here.
    #res_mask = mask_info.split(",")
    print(ss_indices)
    for idx in range(len(ss_indices)): #
        if ss_indices[idx] == '': continue # no secondary structure annotation found
        ss_idx = ss_indices[idx].split(",")
        indices_for_dropout = [b for b in ss_idx]
        #for i in indices_for_dropout:
            #start, end = [int(x) for x in i.split(":")]
        for hot in indices_for_dropout:
            mask[idx,int(hot),:20] = 0 # 0 means this part is given. 

    #mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    #batch["mask_inpaint"] = mask

    #batch["seq_mask"] = seq_mask
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)
    #batch["seq_mask"] = seq_mask.to(device=config.device, dtype=torch.bool)
    return batch

def selected_mask_batch3_3(batch, mask_info, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape

    mask = torch.zeros(B, N)
    seq_mask = torch.zeros(B,N,N).bool()

    ss_indices = mask_info
    #res_mask = mask_info.split(",")
    for idx in range(len(ss_indices)): # 
        if ss_indices[idx] == '': continue # no secondary structure annotation found
        ss_idx = ss_indices[idx].split(",")
        indices_for_dropout = [b for b in ss_idx]
        #for i in indices_for_dropout:
            #start, end = [int(x) for x in i.split(":")]
        for r in indices_for_dropout:
            if ":" in r:
                start_idx, end_idx = r.split(":")
                mask[idx, int(start_idx):int(end_idx)] = 1

                seq_mask[idx,int(start_idx):int(end_idx),mid_ptr-10:mid_ptr+10] = True # in dimetion B, N, N # mid point update into 90 instead of 64 due to the dimention changed.
            else:
                mask[idx,int(r)] = 1

    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
    #batch["mask_inpaint"] = mask

    #batch["seq_mask"] = seq_mask
    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)
    batch["seq_mask"] = seq_mask.to(device=config.device, dtype=torch.bool)
    return batch



# need to update on this function too to be able to do sampling. Feb 11 2023
# the previous update is for antibody sgm.  I updated it again on May 25 2023 to make it work on helixsgm model.
# this time, the previous version would not be saved. pleasae see antibody-sgm or helixsgm (not inpainting version) for the previous implementations.
# assuming mask_info "1,4,6"
# I also need to update selected_mask_batch3 and random_mask_batch for it to be working.

def get_condition_from_batch(config, batch, mask_info=None):
    batch_size = batch["coords_6d"].shape[0]
    out = {}
    for c in config.model.condition:
        if c == "length":
            lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]
            mask = torch.zeros(batch_size, config.data.max_res_num,
                               config.data.max_res_num).bool()  # B, N, N
            for idx, l in enumerate(lengths):
                mask[idx, :l, :l] = True
            out[c] = mask
        elif c == "ss":
            out[c] = batch["coords_6d"][:,4:7]
        elif c == "inpainting":
            if mask_info is not None:
                batch_masked = selected_mask_batch3(batch,mask_info,config) # updated for batch to batch 3 function name for inpainting.
            else:
                batch_masked = random_mask_given_all_except_3hotspots_batch(batch, config)
                #batch_masked["seq_mask"] = None

            out[c] = {
                "coords_6d": batch_masked["coords_6d"],
                "mask_inpaint": batch_masked["mask_inpaint"]
               # "seq_mask": batch_masked["seq_mask"]
            }

    return recursive_to(out, config.device)


def get_condition_from_batch_hot4(config, batch, mask_info=None):
    batch_size = batch["coords_6d"].shape[0]
    out = {}
    for c in config.model.condition:
        if c == "length":
            lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]
            mask = torch.zeros(batch_size, config.data.max_res_num,
                               config.data.max_res_num).bool()  # B, N, N
            for idx, l in enumerate(lengths):
                mask[idx, :l, :l] = True
            out[c] = mask
        elif c == "ss":
            out[c] = batch["coords_6d"][:,4:7]
        elif c == "inpainting":
            if mask_info is not None:
                batch_masked = selected_mask_batch3(batch,mask_info,config) # updated for batch to batch 3 function name for inpainting.
            else:
                batch_masked = random_mask_given_all_except_4hotspots_batch(batch, config)
                #batch_masked["seq_mask"] = None

            out[c] = {
                "coords_6d": batch_masked["coords_6d"],
                "mask_inpaint": batch_masked["mask_inpaint"]
               # "seq_mask": batch_masked["seq_mask"]
            }

    return recursive_to(out, config.device)




def get_conditions_random(config, batch_size=8):
    # Randomly sample pdbs from dataset
    # Load into dataset/loader and extract info: not very elegant
    paths = list(Path(config.data.dataset_path).iterdir())
    selected = np.random.choice(paths, 100, replace=False)
    ss_constraints = True if config.data.num_channels == 8 else False
    ds = ProteinDataset(config.data.dataset_path, config.data.min_res_num,
                             config.data.max_res_num, ss_constraints)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))
    condition = get_condition_from_batch(config, batch)
    return condition

def get_conditions_from_pdb(pdb, config, chain="A", mask_info=None, batch_size=8):
    tempdir = tempfile.TemporaryDirectory()
    # isolate chain
    st = load_structure(pdb)
    st_chain = st[struc.filter_amino_acids(st) & (st.chain_id == chain)]
    save_structure(Path(tempdir.name).joinpath(f"{Path(pdb).stem}_chain_{chain}.pdb"), st_chain)

    ss_constraints = True if config.data.num_channels == 8 else False
    ds = ProteinDataset(tempdir.name, config.data.min_res_num,
                        config.data.max_res_num, ss_constraints)

    dl = torch.utils.data.DataLoader([ds[0]]*batch_size, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))

    return get_condition_from_batch(config, batch, mask_info=mask_info)

def get_conditions_from_protein_dataset(pkl_name, index, config, chain="A", mask_info=None, batch_size=8):
    
    
    with open(pkl_name,"rb") as f:
        ds= pickle.load(f)

    # read pickle
    # the below is for mask_info given into the pickle dataset as a dict. I changed it into mask_info
    #mask_info = [ds[index]["ss_indices"]] * batch_size
    mask_info = [mask_info] * batch_size

    dl = torch.utils.data.DataLoader([ds[index]]*batch_size, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))
    
    return get_condition_from_batch(config, batch, mask_info=mask_info)



def get_mask_all_lengths(config, batch_size=16):
    all_lengths = np.arange(config.data.min_res_num, config.data.max_res_num+1)

    mask = torch.zeros(len(all_lengths), batch_size, config.data.max_res_num,
                       config.data.max_res_num).bool()  # L, B, N, N

    for idx,l in enumerate(all_lengths):
        mask[idx, :, :l, :l] = True

    return mask

def run_tmalign(path1, path2, binary_path="tm/TMalign", fast=True):
    cmd = [binary_path, path1, path2]
    if fast:
        cmd += ["-fast"]
    result = subprocess.run(cmd, capture_output=True)
    result = result.stdout.decode("UTF-8").split("\n")
    if len(result) < 10: return 0. # when TMalign throws error
    tm = result[13].split(" ")[1].strip()
    return float(tm)

def show_all_channels(sample, path=None, nrows=1, ncols=8):
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.1,
                     share_all=True
                     )

    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])

    ax_idx = 0
    for s in sample:
        for ch in range(ncols):
            grid[ax_idx].imshow(s[ch])
            ax_idx += 1

    if path:
        plt.savefig(path)




def random_mask_given_all_except_3hotspots_batch(batch, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    #mask_min = config.model.inpainting.mask_min_len
    #mask_max = config.model.inpainting.mask_max_len

    #random_mask_prob = config.model.inpainting.random_mask_prob
    #contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob

   # lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    #prob = random.random()
    # so one means no masking this position. zero mean masking.
    m = []
    for i in range(B):
        hotspot_3_lst = random.sample(range(14), 3)
        mask = torch.ones(N,N) # ones mean regerterate regions. like the CDR regions. zero means given regions, like FR region. for example :64, 64/the aa part is :14,:20
        for hot in hotspot_3_lst:
            mask[hot,:20] = 0
        m.append(mask)
    mask_d = torch.stack(m, dim=0)

    batch["mask_inpaint"] = mask_d.to(device=config.device, dtype=torch.bool)
    return batch




def random_mask_given_all_except_4hotspots_batch(batch, config):
    if "inpainting" not in config.model.condition:
        batch["mask_inpaint"] = None
        return batch

    B, _, N, _ = batch["coords_6d"].shape
    #mask_min = config.model.inpainting.mask_min_len
    #mask_max = config.model.inpainting.mask_max_len

    #random_mask_prob = config.model.inpainting.random_mask_prob
    #contiguous_mask_prob = config.model.inpainting.contiguous_mask_prob

   # lengths = [len([a for a in i if a != "_"]) for i in batch["aa_str"]]  # get lengths without padding token
    # Decide between none vs random masking vs contiguous masking
    #prob = random.random()
    # so one means no masking this position. zero mean masking.
    m = []
    for i in range(B):
        hotspot_4_lst = random.sample(range(14), 4)
        mask = torch.ones(N,N) # ones mean regerterate regions. like the CDR regions. zero means given regions, like FR region. for example :64, 64/the aa part is :14,:20
        for hot in hotspot_4_lst:
            mask[hot,:20] = 0
        m.append(mask)
    mask_d = torch.stack(m, dim=0)

    batch["mask_inpaint"] = mask_d.to(device=config.device, dtype=torch.bool)
    return batch



#     if prob < random_mask_prob:
#         # Random masking
#         mask = []
#         for l in lengths:
#             rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
#             rand_indices = torch.randperm(l)[:rand]

#             m = torch.zeros(N)
#             m[rand_indices] = 1

#             mask.append(m)
#         mask = torch.stack(mask, dim=0)
#     elif prob > 1-contiguous_mask_prob:
#         # Contiguous masking
#         mask = []
#         for l in lengths:
#             rand = torch.randint(int(mask_min * l), int(mask_max * l), (1,))[0]
#             index = torch.randint(0, (l - rand).int(), (1,))[0]

#             m = torch.zeros(N)
#             m[index:index + rand] = 1

#             mask.append(m)
#         mask = torch.stack(mask, dim=0)
#     else:
#         mask = torch.ones(B, N) # No masking

#    mask = torch.logical_or(mask.unsqueeze(-1), mask.unsqueeze(1)) # B, N -> B, N, N
#    batch["mask_inpaint"] = mask.to(device=config.device, dtype=torch.bool)

#    return batch



# the below function is added on May 31 2023 to let the inpaint program to be able to run with only seq and pos info. no pdb needed.

# the letter order here is only for helix. antibody is different. 

def one_hot_encode(seq):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    mapping = dict(zip(alphabet,range(len(alphabet))))
    seq_idx = [mapping[s] for s in seq]
    return np.eye(len(alphabet))[seq_idx]



def get_conditions_by_specify_hotspots(pkl_name, index, config, chain,batch_size, mask_info, hot_res):


    with open(pkl_name,"rb") as f:
        ds= pickle.load(f)

    # read pickle
    # the below is for mask_info given into the pickle dataset as a dict. I changed it into mask_info
    #mask_info = [ds[index]["ss_indices"]] * batch_size
    hot_pos_lst = mask_info.split(",")
    mask_info = [mask_info] * batch_size
    hot_res_lst = hot_res.split(",")
    for i,pos in enumerate(hot_pos_lst):
        ds[index]["coords_6d"][0,int(pos),:20] = torch.from_numpy(one_hot_encode(hot_res_lst[i]))    # let's assuming the shape is C,N,N


    dl = torch.utils.data.DataLoader([ds[index]]*batch_size, batch_size=batch_size,
                                     collate_fn=PaddingCollate(config.data.max_res_num))
    batch = next(iter(dl))

    return get_condition_from_batch(config, batch, mask_info=mask_info)


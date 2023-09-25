
# compared with sampling_hot3_D_type.py file, it just generated more data. and saved in samples222_ files.
# THIS FILE FOR d TYPE. THE DIFFERENCE IS TO ADD WINDOW SHIFT ON RESIDUE POS 
import torch
from pathlib import Path
from score_sde_pytorch.utils import get_model, restore_checkpoint, recursive_to
from score_sde_pytorch.models.ema import ExponentialMovingAverage
import score_sde_pytorch.sde_lib as sde_lib
import score_sde_pytorch.sampling as sampling
import score_sde_pytorch.losses as losses
import pickle as pkl
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from utils import get_conditions_random, get_mask_all_lengths, get_conditions_from_pdb,get_conditions_from_protein_dataset,get_conditions_by_specify_hotspots
import pickle
from dataset import ProteinDataset, PaddingCollate

#index = 4659
# read pickle file

#helix 1

import numpy as np

target_coords_NC = np.array([[149.347, 135.834, 118.828],
       [149.347, 135.834, 118.828],
       [148.759, 134.506, 118.367],
       [149.026, 134.193, 116.933],
       [150.156, 133.527, 116.514],
       [147.457, 140.27 , 120.406],
       [146.562, 141.112, 124.087],
       [145.805, 140.146, 124.299],
       [151.363, 143.555, 118.29 ],
       [149.965, 144.097, 118.51 ],
       [149.979, 145.342, 119.323],
       [149.775, 145.3  , 120.694],
       [161.306, 148.153, 125.059]], dtype="float32")




# change L type hotspots to D type
target_coords_NC = target_coords_NC * -1
#####target_atom_name = target_atom_names[kkk]
target_atom_name =['CA_HIS_7',
 'CB_HIS_7',
 'CG_HIS_7',
 'ND1_HIS_7',
 'CA_GLU_9',
 'CD_GLU_9',
 'OE1_GLU_9',
 'CA_PHE_12',
 'CB_PHE_12',
 'CG_PHE_12',
 'CD2_PHE_12']


start_pos = 7




import pickle 



d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}



def convert_to_pos_lst_and_res_lst(lst): # lst see target_coords_name_nameNoflat_CAcoords_lst0912 index2
    pos_lst = []
    res_lst = []
    for sub_lst in lst:
        lst_ss = sub_lst[0].split("_")
        pos = int(lst_ss[-1])
        pos_lst.append(str(pos-start_pos))
        res_lst.append(d[lst_ss[-2]])
    return ','.join(pos_lst),','.join(res_lst)




REV_ONE_HOT = 'ACDEFGHIKLMNPQRSTVWY'

#kkk = 0
target_atom_names = [target_atom_name] #test_lst[1]
#from Bio.SeqUtils import seq1, seq3
import itertools

def get_pos_res_id(target_atom_name):
    pos_lst = []
    res_lst = []
    for sub_str in target_atom_name:
        sub_lst = sub_str.split("_")
        res=sub_lst[1]
        pos = int(sub_lst[2])-start_pos
        if pos not in pos_lst:
            pos_lst.append(pos)
            res_lst.append(d[res])
    return pos_lst,','.join(res_lst)

#pos_lst, res_lst =get_pos_res_id(target_atom_names[kkk])

def check_lst(your_list):
    if len(your_list) != len(set(your_list)) or your_list ==[]:
        return False
    for num in your_list:
        num = int(num)
        if num<0 or num>13:
            return False
    return True




def grep_d_resid_pos(index):
# pos need to be 0~13
    pos_lst, res_str =get_pos_res_id(target_atom_names[index])
    all_lst = []
    for pos_num in pos_lst:
        all_lst.append([str(pos_num-1),str(pos_num),str(pos_num+1)]) # this part is wrong. I already delete strt index in get_pos_id.  so no need to do it gain here. udpated this is due to pos_num start 1. but helixsgm start from 0
    comb_pos_lst = list(map(list,list(itertools.product(*all_lst))))
    filter_lst = list(filter(check_lst,comb_pos_lst))
    comb_pos_lst2 = list(map(','.join,filter_lst))
    return res_str,comb_pos_lst2



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--pdb', type=str, default=None)
    parser.add_argument('--chain', type=str, default="A")
    parser.add_argument('--mask_info', type=str, default="1,5,9")
    parser.add_argument('--tag', type=str, default="test3s_ab")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_iter', type=int, default=30)
    parser.add_argument('--select_length', type=bool, default=False)
    parser.add_argument('--length_index', type=int, default=1) # Index starts at 1
    parser.add_argument('--pkl', type=str, default=None) # Index starts at 1
    parser.add_argument('--givenhot', type=int, default=0)
    parser.add_argument('--start_index', type=int, default=0)

 #   parser.add_argument('--singleCDR', type=int, default=4) # 4 means not single. to generated mutiple. 1. cdr 1. 2 .cdr 2 . 3, cdr 3/   
    args = parser.parse_args()
    if args.givenhot == 0:
        givenhot = False
    else:
        givenhot = True
    start_index = args.start_index * 2
#    assert not (args.pdb is not None and args.select_length)

    with open(args.config, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    config.device = args.device
    workdir = Path("sampling", "case_study_glp1_helix1", Path(args.config).stem, Path(args.checkpoint).stem, args.tag)

    # Initialize model.
    score_model = get_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    state = restore_checkpoint(args.checkpoint, state, args.device)
    state['ema'].store(state["model"].parameters())
    state['ema'].copy_to(state["model"].parameters())

    # Load SDE
    if config.training.sde == "vesde":
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    elif config.training.sde == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3

    # Sampling function
    sampling_shape = (args.batch_size, config.data.num_channels,
                      config.data.max_res_num, config.data.max_res_num)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)
    
    if True:
    #for index in range(start_index,start_index+2):
        index = start_index
        generated_samples2 = []
        for j in tqdm(range(args.n_iter)):
            #if args.select_length:
            #    mask = get_mask_all_lengths(config,batch_size=args.batch_size)[args.length_index-1]
            #    condition = {"length": mask.to(config.device)}
        #elif args.pdb is not None:
        #    condition = get_conditions_from_pdb(args.pdb, config, args.chain, args.mask_info, batch_size=args.batch_size)
        # I conmmented it out for simple purpse. I will change to my data later.
            if givenhot:
                #pos_str,res_str = convert_to_pos_lst_and_res_lst(test_lst[2][index])#convert_to_pos_lst_and_res_lst(test_lst[2][0])
                res_str,pos_str_lst = grep_d_resid_pos(index)
                for pos_str in pos_str_lst:
                    condition = get_conditions_by_specify_hotspots(args.pkl, index, config, args.chain,args.batch_size, pos_str,res_str)            
            #elif args.pkl is not None and not givenhot:
            #print(args.pkl, index, config, args.chain, args.mask_info, args.batch_size)
            #    condition = get_conditions_from_protein_dataset(args.pkl, index, config, args.chain, args.mask_info, batch_size=args.batch_size)
        
            #else:
            #    condition = get_conditions_random(config, batch_size=args.batch_size)
                    sample, n = sampling_fn(state["model"], condition)
                    generated_samples2.append(sample.cpu())
        
            generated_samples = torch.cat(generated_samples2, 0)

            workdir.mkdir(parents=True, exist_ok=True)
            with open(workdir.joinpath("sample4_lalala"+str(index)+"_"+str(start_pos)+".pkl"), "wb") as f:
                pkl.dump(generated_samples, f)

if __name__ == "__main__":
    main()

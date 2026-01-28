import argparse
import os
import os.path
import sys
import numpy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import DistanceMetric
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import random
import math
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from Bio import SeqIO
import torch

EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_embeddings=True, return_hidden_states=True
)

#layerindex=[35]

def embed_prot(sequence,model,layerindex,device,EMBEDDING_CONFIG):
    print("Performing embedding with model: "+model+" on device: "+device+", distance analysis on layer: "+str(layerindex))
    protein = ESMProtein(sequence=sequence)
    client = ESMC.from_pretrained(model).to(device) # or "cpu"
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
    hy=logits_output.hidden_states[layerindex,0,:,:].detach().to(torch.float).cpu()
    return hy


def count_aa(aa,aln):
    count=0
    for i in range(len(aln)):
        if aln[i] == aa:
            count=count+1

    return(count)

#count_aa('-',[x[16] for x in alignments])


def generate_dict_aln(file:str,init_seq:str,freq:int):
    alignments = [''.join(seq.split("\n")[1:]) for seq in open(file).read().split(">")[1:]]
    acclist=[(seq.split("\n")[0]) for seq in open(file).read().split(">")[1:]]
    accdict={acclist[i]:i for i in range(len(acclist))}

    #print(acclist)
    if init_seq not in list(accdict.keys()):
        sys.exit('Cannot find init_seq_accession in the alignment file, please check your inputs!')
    
    nracc=accdict[init_seq]
    aln_post=[]

    for i in range(len(alignments[nracc])):
        if(alignments[nracc][i] != '-'):
            aln_post.append(i+1)

    dict_aln={}

    for j in range(len(aln_post)):
        aa_post=[i+1 for i in range(len(alignments[nracc].replace('-','')))][j]
        dict_aln[aa_post] = {
        "aa_type":[alignments[nracc].replace('-','')[i] for i in range(len(alignments[nracc].replace('-','')))][j],
        "aa_post":aa_post,
        "aln_post":aln_post[j]
        }
    
    lenaln=len(alignments)    
    for i in dict_aln.keys():
        p=dict_aln[i]['aln_post']
        aalist={}
        for j in set([x[p-1] for x in alignments]):
            aalist.update({
            j:count_aa(j,[x[p-1] for x in alignments])/lenaln*100
            })
        dict_aln[i].update({
            'aa_perc':aalist
        })
    
    for i in dict_aln.keys():
        sub_aa={}
        for x in dict_aln[i]['aa_perc']:
            if dict_aln[i]['aa_perc'][x] >= freq and x != dict_aln[i]['aa_type']:
                sub_aa.update({x:dict_aln[i]['aa_perc'][x]})
    
        dict_aln[i].update({
            'aa_sub_cand':sub_aa
        })

    return dict_aln

def generate_sub_dict(dict_aln):
    sub_aa_dict={}
    for i in dict_aln.keys():
        if dict_aln[i]['aa_sub_cand'] != {} and list(dict_aln[i]['aa_sub_cand'].keys()) != ['-']:
            sub_aa_dict.update({i:dict_aln[i]})

    sub_dict={}
    print(dict_aln[205])
    for i in sub_aa_dict.keys():
        aalist=list(sub_aa_dict[i]['aa_sub_cand'].keys())
        
        if '-' in aalist:
            aalist.remove('-')
            
        sub_dict.update({sub_aa_dict[i]['aa_post']:aalist})
    
    print(sub_dict)
    return sub_dict

def aa_sub_seq(sub_dict,dict_aln):
    combine_list=[]
    seq=[]
    subdict={}
    for i in sub_dict.keys():
        subdict.update({i:dict_aln[i]['aa_type']})

    acc=1
    for i in sub_dict.keys():
        oriseq=[dict_aln[x]['aa_type'] for x in dict_aln.keys()]
        for p in sub_dict.keys():
            subdict.update({p:dict_aln[p]['aa_type']})
            
        for j in sub_dict[i]:
           # subdict[i]=j
            print(str(i)+'_'+j)
            subdict.update({i:j})
            loc=dict_aln[i]['aa_post']
            print(loc)
            oriseq[loc-1] = j
            combine_list.append({dict_aln[i]['aa_type']+''+str(i)+''+j:[subdict[x] for x in subdict.keys()]})
            seq.append({dict_aln[i]['aa_type']+''+str(i)+''+j:''.join(oriseq)})
           # print(all_combine_dict)
            acc=acc+1


    return(combine_list,seq)
  
def omit_seq_bystep(sub_dict_bystep,sequence,dict_aln):
    oriseq=list(sequence)
    acc=[]
    for i in sub_dict_bystep.keys():
        for j in sub_dict_bystep[i]:
            loc=dict_aln[i]['aa_post']
            oriseq[loc-1] = j
            acc.append(str(i)+j)

    return {'abbr':"_".join(acc),
            'sub_log':sub_dict_bystep,
            'seq':''.join(oriseq)}
            

def get_all_combined(sub_log):
    comlist=[]
    comdict={}
    for i in sub_log.keys():
        for j in sub_log[i]:
            comdict.update({str(i)+j:{i:j}})

    return comdict

#sublog=  [sub_seq_emb[i]['sub_log'] for i in sub_seq_emb.keys()]
def generate_sub_dict_bystep(sub_dict,sub_log):
    last_sub_log=sub_log[-1]
    last_subcom=get_all_combined(last_sub_log)
    all_subcom=get_all_combined(sub_dict)
    allchoices={}
    for i in all_subcom.keys():
        if i not in list(last_subcom.keys()):
            allchoices.update({i:
                               {'loc':''.join([str(k) for k in list(all_subcom[i].keys())]),
                               'aa':''.join([all_subcom[i][k] for k in all_subcom[i].keys()])}})

    lenchoices=len(allchoices)
    if lenchoices <1:
        sys.exit('Number of availible substitutions used up, please choose a smaller number of maximum steps!')
    randchange=random.randint(0, lenchoices-1)
    randchoices=allchoices[list(allchoices.keys())[randchange]]
    newsub={}
    for i in last_sub_log.keys():
        newsub.update({i:last_sub_log[i]})
    
    newsub.update({int(randchoices['loc']):[randchoices['aa']]})
    return(newsub)


def dist_estimator(sub_seq_emb_dict,metric='minkowski'):
    dist = DistanceMetric.get_metric('minkowski')
    lenlist=len(list(sub_seq_emb_dict.keys()))
    Y=torch.vstack(([sub_seq_emb_dict[i]['hidden_state'][0,:,:].mean(-1) for i in list(sub_seq_emb_dict.keys())[:lenlist-1]]))
    X=torch.vstack(([sub_seq_emb_dict[i]['hidden_state'][0,:,:].mean(-1) for i in sub_seq_emb_dict.keys()]))
    return dist.pairwise(X,Y)


def generate_sub_path(num_steps:int,init_seq:str,sub_dict:dict,start_sub:dict,dict_aln:dict,dict_condition:float,step_dist:float,device:str,model:str,layerindex):
    wt_seq={'abbr':'INIT',
        'sub_log':{},
        'seq':init_seq}

    wt_seq.update({'hidden_state':embed_prot(wt_seq['seq'],model,layerindex,device,EMBEDDING_CONFIG)})
    sub_seq_emb={wt_seq['abbr']:wt_seq}
    sublog=  [sub_seq_emb[i]['sub_log'] for i in sub_seq_emb.keys()]
    if start_sub != {}:
        myseq=omit_seq_bystep(start_sub,init_seq,dict_aln)['seq']
        #print(start_sub)
        start_seq={'abbr':'STARTLOC',
        'sub_log':start_sub,
        'seq':myseq}
        start_seq.update({'hidden_state':embed_prot(start_seq['seq'],model,layerindex,device,EMBEDDING_CONFIG)})
        sub_seq_emb.update({start_seq['abbr']:start_seq})
    
    sublog=  [sub_seq_emb[i]['sub_log'] for i in sub_seq_emb.keys()]
    for k in range(num_steps):
        #sublog=  [sub_seq_emb[i]['sub_log'] for i in sub_seq_emb.keys()]
        next_sub_dict=generate_sub_dict_bystep(sub_dict,sublog)
        sub_seq=omit_seq_bystep(next_sub_dict,init_seq,dict_aln)
        sub_seq.update({'hidden_state':embed_prot(sub_seq['seq'],model,layerindex,device,EMBEDDING_CONFIG)})
        sub_seq_emb.update({sub_seq['abbr']:sub_seq})
        print("Completed "+str(k+1)+" rounds of single nucleoside substitution and embedding generation.")
        subloglist=  [sub_seq_emb[i]['sub_log'] for i in sub_seq_emb.keys()]
        print('Minkowski distance matrix at step '+str(k)+': ')
        print(sub_seq['hidden_state'].shape)
        dist_mat=dist_estimator(sub_seq_emb)
        print(dist_mat)
        if dist_mat[-1,:].mean() > dict_condition or abs((dist_mat[-1] - dist_mat[-2])[0]) > step_dist:
            sublog=subloglist[0:len(subloglist)-1]
            print("Back to the last node as distance to far at step " +str(k))
            sub_seq_emb.pop(sub_seq['abbr'])

        else: 
            sublog=subloglist
            print("Continue with the current path at step "+str(k))

    return sub_seq_emb


def plot_path(simu_path_emb):
    arg_dict={}
    acclist=simu_path_emb.keys()
    for i in simu_path_emb.keys():
    #difft=emb[i]['embeddings'][0] - emb['WT.luc_relaxed_complex_A']['embeddings'][0]
        difft=simu_path_emb[i]['hidden_state'][0,:,:]
        arg_dict.update({i:difft})

    diff_dict={}
    for i in acclist:
    #difft=emb[i]['embeddings'][0] - emb['WT.luc_relaxed_complex_A']['embeddings'][0]
        difft=torch.vstack(([arg_dict[x]-simu_path_emb[i]['hidden_state'][0,:,:] for x in arg_dict.keys()]))
        diff_dict.update({i:difft})


    clusterdf=torch.vstack(([diff_dict[x].mean(-1)  for x in diff_dict.keys()]))
    #clusterdf=torch.vstack(([arg_dict[i] for i in arg_dict.keys()]))
    pca = PCA(n_components=2)
    pca.fit(clusterdf)

    projected_mean_embeddings = pca.transform(clusterdf)
    #acc=[simu_path_emb[i]['abbr'] for i in simu_path_emb.keys()]
    acc=[i for i in range(len(simu_path_emb.keys()))]
    subinfo=[simu_path_emb[i]['abbr'] for i in simu_path_emb.keys()]
    df=pd.DataFrame({
        'Step':acc,
        'Sub_info':subinfo,
        'PC1':projected_mean_embeddings[:,0],
        'PC2':projected_mean_embeddings[:,1]
    })
    # x and y given as array_like objects
    fig = px.scatter(df,x='PC1', y='PC2',text='Step',hover_data='Sub_info')
    return fig


def write_output(out_dir:str,simu_path_emb):
    fin=[simu_path_emb[i] for i in simu_path_emb.keys()][-1]
    f=open(out_dir+'/path.'+fin['abbr']+'.pkl','wb')
    pickle.dump(simu_path_emb,f)
    f = open(out_dir+'/'+fin['abbr']+'.faa', 'a')
    seq=fin['seq']
    f.writelines(['>'+ fin['abbr'], '\n',seq,'\n'])
    f.close()
    path_fig=plot_path(simu_path_emb)
    path_fig.write_html(out_dir+'/path.'+fin['abbr']+'.html')

def warp_start_loc_todict(start_loc,mmseq):
    start_locdict={}
    if ';' in list(start_loc):
        start_loclist=start_loc.split(';')
        if len(start_loclist) >0:
            for item in start_loclist:
                if ':' in list(item):
                    item=item.split(':')
                    if item[1] in ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']:
                        if int(item[0]) <= len(mmseq):
                            start_locdict.update({int(item[0]):[item[1]]})

    return(start_locdict)

def start_simulation(aln_file:str,init_seq_acc:str,freq:int,num_steps:int,out_dir:str,start_loc_str,dict_condition,step_dist,device,model,layerindex):
    dict_aln=generate_dict_aln(aln_file,init_seq_acc,freq)
    sub_dict=generate_sub_dict(dict_aln)
    mmseq=''.join([dict_aln[x]['aa_type'] for x in dict_aln.keys()])
    start_loc=warp_start_loc_todict(start_loc_str,mmseq)
    simu_path_emb=generate_sub_path(num_steps,mmseq,sub_dict,start_loc,dict_aln,dict_condition,step_dist,device,model,layerindex)
    write_output(out_dir,simu_path_emb)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-alignment", help="path to the protein alignment file.", type=str, required=True)
    parser.add_argument("-init_seq_accession", help="the accession to the sequence, on which locus refinement will be performed.", type=str, required=True)
    parser.add_argument("-frequency", help="the frequency in percentage of the loci to perform refinement, the value should be an integer from 0 to 100.", type=int, default=40)
    parser.add_argument("-maximum_num_steps", help="the maximum number of steps of random substitution on candidant loci simulation, note that this may not equal to the final number of substitution performed.", type=int, default=20)
    parser.add_argument("-output_dir", help="the directory to write output files including the refined sequence.", type=str, default='loci_refine_out')
    parser.add_argument("-start_loc", help="You can start refinement with a few preset substitutions, please wrap them in a string such as 1:C;4:F", type=str, default='')
    parser.add_argument("-maxi_mean_distance", help="the maximum distance of a new simulation to all other in the path.", type=float, default=4)
    parser.add_argument("-maxi_bystep_distance", help="the maximum distance of a new simulation to the last one in the path.", type=float, default=2)
    parser.add_argument("-device", help="device to perform llm embedding on, cpu or cuda", type=str, default='cpu')
    parser.add_argument("-model", help="ESM-C model to perform embedding, you may choose from esmc_300m, esmc_600m and esmc_6b, please note that you may need networks to download weights", type=str, default='esmc_300m')
    parser.add_argument("-layer", help="The layer of hidden states to perform distance analysis",type=int, default=29)
    args = parser.parse_args()

    aln_file = args.alignment
    init_seq_acc = args.init_seq_accession
    freq = args.frequency
    num_steps = args.maximum_num_steps
    out_dir = args.output_dir
    start_loc_str = args.start_loc
    dict_condition = float(args.maxi_mean_distance)
    step_dist=float(args.maxi_bystep_distance)
    device = args.device
    model = args.model
    layerindex=[int(args.layer)]

    if not os.path.isfile(aln_file):
        sys.exit('The protein alignment file does not exist!')

    if model not in ['esmc_300m','esmc_600m','esmc_6b']:
        sys.exit('ESM-C model name not recognized, please check your -model input!')
    elif model == 'esmc_300m':
        if int(args.layer) > 29:
            sys.exit('The model layer input exceed the maximum number of ' + model+ ' layers, please choose a number less than 30')
    elif model == 'esmc_600m':
        if int(args.layer) > 35:
            sys.exit('The model layer input exceed the maximum number of ' + model+ ' layers, please choose a number less than 35')
    elif model == 'esmc_6b':
        if int(args.layer) > 79:
            sys.exit('The model layer input exceed the maximum number of ' + model+ ' layers, please choose a number less than 79')
    
    out_dir = os.getcwd() + '/' + args.output_dir
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    start_simulation(aln_file,init_seq_acc,freq,num_steps,out_dir,start_loc_str,dict_condition,step_dist,device,model,layerindex)
    
if __name__ == '__main__':
	main()


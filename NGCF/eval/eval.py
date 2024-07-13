import pickle
import torch

from utility.eval_elliptic import *
import pandas as pd
from NGCF import NGCF
import numpy as np
import matplotlib.pyplot as plt


def load_dg_args(model_path):
    print(f"Loading Data Generator")
    model_arg = model_path + '/eval/data_generator_args.pkl'
    with open(model_arg, 'rb') as f:
        data_generator, args = pickle.load(f)
    data_generator.path = "../"+ data_generator.path
    return data_generator, args

def load_model(model_path,data_generator, args):
    print(f"Loading Weights ")
    weights = torch.load(model_path + '/best_model.pkl')
    print(f"Loading Adj Mat")
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)
    model.load_state_dict(weights)
    return model, data_generator, args

def load_model_and_dg_args(model_path):
    data_generator, args = load_dg_args(model_path)
    model, data_generator, args = load_model(model_path,data_generator, args)
    return model, data_generator, args
def change_abk(a,b,k,args):
    args.a = a
    args.b = b
    args.k = k
    return args


def run_single_exp_n_times(model,data_generator,args,exp,n_times=5):
    args = change_abk(*exp,args)
    hr_list, ngcf_list = [], []
    for i in range(n_times):
        hr, ngcf = eval_elliptic(data_generator, model,args)
        print(f"Evaluation {i}HR: {hr}, NGCF: {ngcf} ")
        hr_list.append(hr)
        ngcf_list.append(ngcf)
    print("**************")
    return np.mean(hr_list), np.mean(ngcf_list)

def plot_results(hr_list, ngcf_list):
    plt.plot(hr_list)
    plt.plot(ngcf_list)
    plt.show()

def run_exps(model, data_generator, args, exps, n_times=5,plot=True):
    hr_list, ngcf_list = [], []
    for i,exp in tqdm(enumerate(exps)):
        print("**************")
        print(f"Running Experiment {i}")
        hr, ngcf = run_single_exp_n_times(model, data_generator, args, exp, n_times)
        hr_list.append(hr)
        ngcf_list.append(ngcf)
    if plot:
        plot_results(hr_list, ngcf_list)
    return hr_list, ngcf_list

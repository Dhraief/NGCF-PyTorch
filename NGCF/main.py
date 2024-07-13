'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
from utility.eval_elliptic import *
from utility.logger import setup_logger
import warnings
import wandb
import yaml
warnings.filterwarnings('ignore')
from time import time

logger = setup_logger(__name__)

path_cfg = "/skunk-pod-storage-mohamed-2eali-2edhraief-40ibm-2ecom-pvc/NGCF-PyTorch/NGCF/utility/cfg.json"


def load_config(args, file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    for key, value in config.items():
        setattr(args, key, value)

args = parse_args()
load_config(args, path_cfg)

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Parameter {name} has no gradient.")
        else:
            print(f"Parameter {name} gradient: {param.grad.norm()}")

def save_data_generator_args(data_generator, args,path):
    if not os.path.exists(f"{path}/eval"):
        os.makedirs(f"{path}/eval")
    with open(f"{path}/eval/data_generator_args.pkl", 'wb') as f:
        
        pickle.dump((data_generator, args), f)
def main():
    wandb.init(project="ngcf")
    wandb.config.update(args, allow_val_change=True)

    args.device = torch.device('cuda:' + str(args.gpu_id))
    logger.info("Launching Data Generation")
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    
    if isinstance(args.node_dropout, str):
        args.node_dropout = eval(args.node_dropout)
    if isinstance(args.mess_dropout, str):
        args.mess_dropout = eval(args.mess_dropout)
    
    prefix_feature = "with_feature" if args.use_features else "no_feature"
    model_path = f"{args.weights_path}{args.dataset}/{prefix_feature}/bs{args.batch_size}_lr{args.lr}e{args.epoch}_nneg{args.n_neg}_r{args.regs}_drop_{args.node_dropout}/"

    save_data_generator_args(data_generator, args,model_path)
    print(data_generator.n_users, data_generator.n_items)
    
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    best_model_weights = None
    best_hr_20 = 0.0
    
    
    weights_dir = os.path.dirname(model_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)


    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            #print(len(users), len(pos_items), len(neg_items))
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           drop_flag=args.node_dropout_flag)
            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            # check_gradients(model)
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
        wandb.log({"train/loss": loss, "train/mf_loss": mf_loss, "train/emb_loss": emb_loss})

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test, drop_flag=False)
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])
        Ks = eval(args.Ks)
        for i in range(len(Ks)):
            wandb.log({f"test/recall_{Ks[i]}": ret['recall'][i],
                       f"test/precision_{Ks[i]}": ret['precision'][i], 
                       f"test/hit_{Ks[i]}": ret['hit_ratio'][i], 
                       f"test/ndcg_{Ks[i]}": ret['ndcg'][i]})
        if ret['hit_ratio'][0] >= best_hr_20:
            best_hr_20 = ret['hit_ratio'][0]
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, model_path + 'best_model.pkl')
            print('save the best weights in path: ', model_path + '_best_model.pkl')
            print("best hr_20 ", best_hr_20)

        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f] best_hr=[%.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, best_hr_20)
                print(perf_str)
            continue

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
        print()
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=20)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            
            torch.save(model.state_dict(), model_path+ str(epoch) + '.pkl')
            print('save the weights in path: ', model_path + str(epoch) + '.pkl')
            # Save the best model weights


    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_hit_0 = max(hit[:, 0])
    idx = list(hit[:, 0]).index(best_hit_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    
    print(final_perf)

    for i in range(len(Ks)):
        wandb.log({f"final/recall_{Ks[i]}": recs[idx][i],
                     f"final/precision_{Ks[i]}": pres[idx][i],
                        f"final/hit_{Ks[i]}": hit[idx][i],
                        f"final/ndcg_{Ks[i]}": ndcgs[idx][i]})
        
    wandb.summary[f"best_rec_{Ks[0]}"] = recs[idx][0]
    wandb.summary[f"best_prec_{Ks[0]}"] = pres[idx][0]
    wandb.summary[f"best_hit_{Ks[0]}"] = hit[idx][0]
    wandb.summary[f"best_ndcg_{Ks[0]}"] = ndcgs[idx][0]
    wandb.summary["epoch"] = idx


    if args.dataset=="elliptic":
        if args.use_features:
            with open("dg_model_args_with_features.pkl", 'wb') as f:
                pickle.dump((data_generator, model, args), f)
        else:
            with open("dg_model_args_no_features.pkl", 'wb') as f:
                pickle.dump((data_generator, model, args), f)
        eval_elliptic(data_generator, model, args)

def sweep():
    with open(f'sweep/{args.sweep_file}', 'r', encoding='utf-8') as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project='ngcf')
    count = 30 if args.sweep_file == "elliptic.yaml" else 10 
    wandb.agent(sweep_id, function=main, count=count)
    
if __name__ == '__main__':
    if args.command == 'run':
        main()
    else: 
        sweep()    
    
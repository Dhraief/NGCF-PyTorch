import torch 
import pickle 
import json 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


GLOBAL_DATA_PATH = "/skunk-pod-storage-mohamed-2eali-2edhraief-40ibm-2ecom-pvc/elliptic-subgraph_v0/"
GLOBAL_OUTPUT_DATA = "/skunk-pod-storage-mohamed-2eali-2edhraief-40ibm-2ecom-pvc/NGCF-PyTorch/Data/elliptic/"
def load_raw_data(cfg):
    print("Loading raw data...")
    data_df = pickle.load(open(f'{GLOBAL_DATA_PATH}{cfg["paths"]["data_df_path"]}', 'rb'))
    print("Loading node_idx_map...")
    node_idx_map = torch.load(f'{GLOBAL_DATA_PATH}{cfg["paths"]["node_idx_map_path"]}')
    print("Loading emb...")
    emb = torch.load(f'{GLOBAL_DATA_PATH}{cfg["paths"]["raw_data_path"]}')[node_idx_map]
    return data_df, emb

def preprocess_data_df(data_df):
    cols_to_keep = ["receivers_mapped","senders_mapped","subg","split","labels","senders_len","receivers_len"]
    data_df = data_df[cols_to_keep]
    return data_df
def generate_train_test(data_df, cfg):
    def generate_cleaned_1_1_ill(data_df):
        data_df_1_1 = data_df[(data_df.senders_len==1) & (data_df.receivers_len==1)]
        data_df_1_1['senders_mapped'] = data_df_1_1['senders_mapped'].apply(lambda x: list(x)[0])
        data_df_1_1['receivers_mapped'] = data_df_1_1['receivers_mapped'].apply(lambda x: list(x)[0])
        return data_df_1_1[data_df_1_1.labels==1]
    def remap(df,col):
        nodes = list(df[col].unique())
        node_map = {node:i for i,node in enumerate(nodes)}
        assert len(node_map) == max(node_map.values())+1
        df[f"{col}_ngcf"] = df[col].apply(lambda x: node_map[x])
        inv_map = {v:k for k,v in node_map.items()}
        return node_map
    def split_data(data_df, cfg):
        data_df_1_1_ill.split = data_df_1_1_ill.split.map({'TRN':'TRN','VAL':'TRN','TST':'TST'})
        trn_df = data_df_1_1_ill[(data_df_1_1_ill.split=='TRN') & (data_df_1_1_ill.labels==1)]
        tst_df = data_df_1_1_ill[(data_df_1_1_ill.split=='TST') & (data_df_1_1_ill.labels==1)]
        return trn_df ,tst_df

    
    data_df_1_1_ill = generate_cleaned_1_1_ill(data_df)
    print("Number of Receivers: " ,data_df_1_1_ill.receivers_mapped.nunique())
    print("Number of Senders: " ,data_df_1_1_ill.senders_mapped.nunique())
    mapping_receivers = remap(data_df_1_1_ill,'receivers_mapped')
    mapping_senders = remap(data_df_1_1_ill,'senders_mapped')
    trn_df, tst_df = split_data(data_df_1_1_ill, cfg)
    return trn_df, tst_df, mapping_receivers,mapping_senders

def get_receivers_to_senders_list(df):
    def get_number_all_senders(senders_list):
        return len(set([sender for senders in senders_list for sender in senders]))
    res_df = df.groupby("receivers_mapped_ngcf")["senders_mapped_ngcf"].apply(set).reset_index()
    receivers, senders_list = res_df.receivers_mapped_ngcf.tolist(), res_df.senders_mapped_ngcf.tolist()
    assert len(receivers) == len(senders_list)
    # get the number of all senders
    print("Number of all senders:", get_number_all_senders(senders_list))
    print("Number of all receivers:", len(receivers))
    return receivers, senders_list 


def create_txt_file(receivers, senders_list, file_path):
    with open(file_path, 'w') as f:
        for receiver,senders  in zip(receivers, senders_list):
            senders_str = ' '.join(map(str, senders))
            f.write(f"{receiver} {senders_str}\n")

def generate_emb_from_map(emb,map):
    inv_map = {v:k for k,v in map.items()}
    return emb[[inv_map[i] for i in range(len(map))]]

def generate_emb(emb, map, path):
    def scale_to_range(tensor, min_val=-1, max_val=1):
        min_tensor = tensor.min()
        max_tensor = tensor.max()
        
        # Scale the tensor to [0, 1]
        tensor_scaled = (tensor - min_tensor) / (max_tensor - min_tensor)
        
        # Shift and scale to [min_val, max_val]
        tensor_scaled = tensor_scaled * (max_val - min_val) + min_val
        return tensor_scaled

    inv_map = {v:k for k,v in map.items()}
    res = emb[[inv_map[i] for i in range(len(map))]]
    assert res.shape == (len(map), emb.shape[1])
    # Standardize the embeddings
    res = scaler.fit_transform(res)
    res = torch.tensor(res, dtype=torch.float32)
    res = scale_to_range(res)
    torch.save(res, path)
    return res

def augment_with_negatives(data_df, map,col):
    assert col in ["senders","receivers"]
    print("Original Length of Senders:", len(map))
    original_length = len(map)

    data_df_lic = data_df[data_df.labels==0]
    lic_list = data_df_lic[f"{col}_mapped"].explode().unique()
    neg_val = set()
    print(f"Added {col}: {len(lic_list)}")
    n = 0 
    for i,val in enumerate(lic_list):
        if val in map.keys(): #already exist in the map
            continue

        map[val] = n+original_length
        neg_val.add(n+original_length)
        n+=1

    print("New Length of Senders:", len(map))
    print(max(map.values()),len(map)-1)
    assert original_length + len(neg_val) == len(map)
    return map,neg_val

def generate_lic_datalist(data_df, mapping_receivers, mapping_senders):
    lic_receivers = data_df[data_df.labels==0].receivers_mapped.tolist()
    lic_senders = data_df[data_df.labels==0].senders_mapped.tolist()
    res = []
    for receivers,senders in zip(lic_receivers,lic_senders):
        #FIXME
        for receiver in receivers:
            for sender in senders:
                res.append((mapping_receivers[receiver],mapping_senders[sender]))
    return res

        
def main():
    with open("config.json", 'r') as file:
        cfg = json.load(file)

    data_df, emb = load_raw_data(cfg)   
    data_df = preprocess_data_df(data_df)
    print("Generating Train Test")
    train_df, test_df, mapping_receivers,mapping_senders = generate_train_test(data_df, cfg)

    print("Generating Receivers and Senders List")
    tr_receivers, tr_senders_list = get_receivers_to_senders_list(train_df)
    tst_receivers, tst_senders_list = get_receivers_to_senders_list(test_df)

    print("Creating Train and Test Files")
    create_txt_file(tr_receivers, tr_senders_list, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['train_file']}")
    create_txt_file(tst_receivers, tst_senders_list, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['test_file']}")

    print("Generating Embeddings")
    mapping_receivers,neg_rec = augment_with_negatives(data_df, mapping_receivers,"receivers")
    emb_receiver = generate_emb(emb, mapping_receivers, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['emb_receivers']}")
    print("Shape of Emb Receivers", emb_receiver.shape)

    mapping_senders,neg_senders = augment_with_negatives(data_df, mapping_senders,"senders")
    emb_senders = generate_emb(emb, mapping_senders, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['emb_senders']}")
    print("Shape of Emb Senders", emb_senders.shape)

    with open(f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['neg_senders']}", 'wb') as f:
        pickle.dump((neg_rec,neg_senders), f)

    data_licit = generate_lic_datalist(data_df, mapping_receivers, mapping_senders)
    with open(f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['path_licit_list']}", 'wb') as f:
        pickle.dump(data_licit, f)


if __name__ == '__main__':
    main()




# ###
# import torch 
# import pickle 
# import json 
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()


# GLOBAL_DATA_PATH = "/skunk-pod-storage-mohamed-2eali-2edhraief-40ibm-2ecom-pvc/elliptic-subgraph_v0/"
# GLOBAL_OUTPUT_DATA = "/skunk-pod-storage-mohamed-2eali-2edhraief-40ibm-2ecom-pvc/NGCF-PyTorch/Data/elliptic/"
# def load_raw_data(cfg):
#     print("Loading raw data...")
#     data_df = pickle.load(open(f'{GLOBAL_DATA_PATH}{cfg["paths"]["data_df_path"]}', 'rb'))
#     print("Loading node_idx_map...")
#     node_idx_map = torch.load(f'{GLOBAL_DATA_PATH}{cfg["paths"]["node_idx_map_path"]}')
#     print("Loading emb...")
#     emb = torch.load(f'{GLOBAL_DATA_PATH}{cfg["paths"]["raw_data_path"]}')[node_idx_map]
#     return data_df, emb

# def preprocess_data_df(data_df):
#     cols_to_keep = ["receivers_mapped","senders_mapped","subg","split","labels","senders_len","receivers_len"]
#     data_df = data_df[cols_to_keep]
#     return data_df
# def generate_train_test(data_df, cfg):
#     def generate_cleaned_1_1_ill(data_df):
#         data_df_1_1 = data_df[(data_df.senders_len==1) & (data_df.receivers_len==1)]
#         data_df_1_1['senders_mapped'] = data_df_1_1['senders_mapped'].apply(lambda x: list(x)[0])
#         data_df_1_1['receivers_mapped'] = data_df_1_1['receivers_mapped'].apply(lambda x: list(x)[0])
#         return data_df_1_1[data_df_1_1.labels==1]
#     def remap(df,col):
#         nodes = list(df[col].unique())
#         node_map = {node:i for i,node in enumerate(nodes)}
#         assert len(node_map) == max(node_map.values())+1
#         df[f"{col}_ngcf"] = df[col].apply(lambda x: node_map[x])
#         inv_map = {v:k for k,v in node_map.items()}
#         return node_map
#     def split_data(data_df, cfg):
#         data_df_1_1_ill.split = data_df_1_1_ill.split.map({'TRN':'TRN','VAL':'TRN','TST':'TST'})
#         trn_df = data_df_1_1_ill[(data_df_1_1_ill.split=='TRN') & (data_df_1_1_ill.labels==1)]
#         tst_df = data_df_1_1_ill[(data_df_1_1_ill.split=='TST') & (data_df_1_1_ill.labels==1)]
#         return trn_df ,tst_df

    
#     data_df_1_1_ill = generate_cleaned_1_1_ill(data_df)
#     print("Number of Receivers: " ,data_df_1_1_ill.receivers_mapped.nunique())
#     print("Number of Senders: " ,data_df_1_1_ill.senders_mapped.nunique())
#     mapping_receivers = remap(data_df_1_1_ill,'receivers_mapped')
#     mapping_senders = remap(data_df_1_1_ill,'senders_mapped')
#     trn_df, tst_df = split_data(data_df_1_1_ill, cfg)
#     return trn_df, tst_df, mapping_receivers,mapping_senders

# def get_receivers_to_senders_list(df):
#     def get_number_all_senders(senders_list):
#         return len(set([sender for senders in senders_list for sender in senders]))
#     res_df = df.groupby("receivers_mapped_ngcf")["senders_mapped_ngcf"].apply(set).reset_index()
#     receivers, senders_list = res_df.receivers_mapped_ngcf.tolist(), res_df.senders_mapped_ngcf.tolist()
#     assert len(receivers) == len(senders_list)
#     # get the number of all senders
#     print("Number of all senders:", get_number_all_senders(senders_list))
#     print("Number of all receivers:", len(receivers))
#     return receivers, senders_list 


# def create_txt_file(receivers, senders_list, file_path):
#     with open(file_path, 'w') as f:
#         for receiver,senders  in zip(receivers, senders_list):
#             senders_str = ' '.join(map(str, senders))
#             f.write(f"{receiver} {senders_str}\n")

# def generate_emb_from_map(emb,map):
#     inv_map = {v:k for k,v in map.items()}
#     return emb[[inv_map[i] for i in range(len(map))]]

# def generate_emb(emb, map, path):
#     def scale_to_range(tensor, min_val=-1, max_val=1):
#         min_tensor = tensor.min()
#         max_tensor = tensor.max()
        
#         # Scale the tensor to [0, 1]
#         tensor_scaled = (tensor - min_tensor) / (max_tensor - min_tensor)
        
#         # Shift and scale to [min_val, max_val]
#         tensor_scaled = tensor_scaled * (max_val - min_val) + min_val
#         return tensor_scaled

#     inv_map = {v:k for k,v in map.items()}
#     res = emb[[inv_map[i] for i in range(len(map))]]
#     assert res.shape == (len(map), emb.shape[1])
#     # Standardize the embeddings
#     res = scaler.fit_transform(res)
#     res = torch.tensor(res, dtype=torch.float32)
#     res = scale_to_range(res)
#     torch.save(res, path)
#     return res

# def augment_senders_with_negatives(data_df, mapping_senders):
#     print("Original Length of Senders:", len(mapping_senders))
#     original_length = len(mapping_senders)
#     data_df_lic = data_df[data_df.labels==0 & (data_df.senders_len==1) & (data_df.receivers_len==1)]
#     lic_senders = data_df_lic.senders_mapped.explode().unique()
#     neg_senders = set()
#     print("Added Senders:", len(lic_senders))
#     n = 0 
#     for i,sender in enumerate(lic_senders):
#         if sender in mapping_senders.keys():
#             continue
#         mapping_senders[sender] = n+original_length
#         neg_senders.add(n+original_length)
#         n+=1
#     print("New Length of Senders:", len(mapping_senders))
#     print(max(mapping_senders.values()),len(mapping_senders)-1)
#     assert original_length + len(neg_senders) == len(mapping_senders)
#     return mapping_senders,neg_senders

# def main():
#     with open("config.json", 'r') as file:
#         cfg = json.load(file)

#     data_df, emb = load_raw_data(cfg)   
#     data_df = preprocess_data_df(data_df)
#     print("Generating Train Test")
#     train_df, test_df, mapping_receivers,mapping_senders = generate_train_test(data_df, cfg)
#     print("Generating Receivers and Senders List")
#     tr_receivers, tr_senders_list = get_receivers_to_senders_list(train_df)

#     tst_receivers, tst_senders_list = get_receivers_to_senders_list(test_df)

#     print("Creating Train and Test Files")
#     create_txt_file(tr_receivers, tr_senders_list, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['train_file']}")
#     print("Creating Test File")
#     create_txt_file(tst_receivers, tst_senders_list, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['test_file']}")

#     print("Generating Embeddings")
#     emb_receiver = generate_emb(emb, mapping_receivers, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['emb_receivers']}")
#     print("Shape of Emb Receivers", emb_receiver.shape)
#     mapping_senders,neg_senders = augment_senders_with_negatives(data_df, mapping_senders)
#     emb_senders = generate_emb(emb, mapping_senders, f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['emb_senders']}")
#     print("Shape of Emb Senders", emb_senders.shape)

#     with open(f"{GLOBAL_OUTPUT_DATA}{cfg['output_paths']['neg_senders']}", 'wb') as f:
#         pickle.dump(neg_senders, f)


# if __name__ == '__main__':
#     main()
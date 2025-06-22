from utils.Utils import Utils as utils
from data.dataset.MultiNodeDataset import MultiNodeDataset
import torch
from datetime import datetime
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from LitModule import LitModule
from litghtningCallback import MyCallback
from models.AttenGRU import AttGRU
import numpy as np
import pandas as pd

# format time

MODEL_LIST = {
    'AttGRU': AttGRU
}
Dataset_dict = {
    'TE': MultiNodeDataset,
    # 'multi': MultiNodeDataset,
    'DC': MultiNodeDataset, 
}

def main():
    args = utils.get_args()
    d_args = vars(args)
    
    multi_target = [t for t in args.target.split(',')]
    if args.time == 'None':
        now = datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        now = args.time
    
    utils.seed_torch(args.seed)
    DEVICE = args.device
    print('==========  Using device: {} for training. ============='.format(DEVICE))
    
    # load the global adjacency matrix
    if 'v10' in args.target:     # 加载田纳西过程的邻接矩阵
        adj = utils.get_global_adj(**d_args)
    else: # DC 过程邻接矩阵
        adj = pd.read_excel('./data/DC_adj.xlsx', index_col=0)
        adj = adj.values
        adj = np.nan_to_num(adj)
        # 去自环
        adj = np.ones_like(adj)
        adj = torch.from_numpy(adj).float().to(DEVICE)
        adj = adj - torch.diag(torch.diag(adj))
    if args.model_type in ['stgcn', 'gcn']:
        adj = utils.get_normalized_adj(adj)
    d_args['node_num'] = adj.shape[0]
    print(adj.shape)
    
    # load data
    cust_data = Dataset_dict[args.dataset](adj=adj, **d_args)
    train_dl, val_dl, test_dl = cust_data.get_dataLoader() 
    scalar = cust_data.get_scaler()

    # Model parameters:
    assert args.model_type in MODEL_LIST, "model type error"

    if args.model_type == 'dcrnn':
        model = MODEL_LIST[args.model_type](adj, **d_args)
    else:
        model = MODEL_LIST[args.model_type](**d_args)
    
    if type(cust_data) == MultiNodeDataset:

        targets_name, targets_idx = cust_data.get_multi_idx()
        litModule = LitModule(model=model, targets_name=targets_name,
                              targets_idx=targets_idx,
                              target_scaler=scalar,
                              **d_args)
    else:
        litModule = LitModule(model=model,  **d_args)
    litModule.set_gloabl_adj(adj)
    trainer = L.Trainer(

        max_epochs=args.epoches,
        default_root_dir='./lightning_logs/' + now + '_' + args.model_type,
        callbacks=[
            # EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode='min'),
            MyCallback()
            ],
        # precision="64-true", #"16-mixed",
        )
    
    # need train loss/ valid loss
    trainer.fit(model=litModule, train_dataloaders=train_dl,val_dataloaders=val_dl)
    train_loss, valid_loss = litModule.get_loss()

    # get the train pred and test pred results with real value
    train_pred = trainer.predict(dataloaders=train_dl)
    test_pred = trainer.predict(dataloaders=test_dl) 
    
    total_train_pred = utils.flattenBatch([batch[0] for batch in train_pred])
    total_train_real = utils.flattenBatch([batch[1] for batch in train_pred])
    total_test_pred = utils.flattenBatch([batch[0] for batch in test_pred])
    total_test_real = utils.flattenBatch([batch[1] for batch in test_pred])

    Ap_atten, Ad = [], []
    if args.model_type == 'AttGRU':
        for i in range(len(test_pred)):
            Ap_atten.extend(test_pred[i][3])
            Ad.extend(test_pred[i][2])
        Ap_atten = np.array([batch[3] for batch in test_pred[:-1]]).reshape(-1, args.node_num, args.node_num)
        Ad = np.array([batch[2] for batch in test_pred[:-1]]).reshape(-1, args.node_num, args.node_num)
        # Ap_atten = np.array([batch[3] for batch in train_pred[:-1]]).reshape(-1, args.node_num, args.node_num)
        torch.cuda.empty_cache()
    
    # save results
    # train_loss, valid_loss = 0, 0
    utils.save_results(train_loss, valid_loss, total_test_pred, total_test_real, args, now=now,
                       train_pred=total_train_pred, train_real=total_train_real,
                       Ap_att=Ap_atten, 
                       Ad=Ad,
                       )

if __name__ == "__main__": 
    main()
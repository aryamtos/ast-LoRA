import torch
from torch.optim import AdamW
from src.ast_LoRA import AST_LoRA, AST_LoRA_ablation

from dataset import Spotify
from utils.training import eval_one_epoch, train_one_epoch

from torch.utils.data import DataLoader
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import datetime
import yaml
import os
import copy


def get_args_parser():

    parser = argparse.ArgumentParser('Lora Transfer-Learning of AST', add_help=False)

    parser.add_argument('--data_train',type=str,help='Path to the location of Train the dataset.')
    parser.add_argument('--data_val',type=str,help='Path to the location of Validation the dataset.')
    parser.add_argument('--data_test',type=str,help='Path to the location of Test the dataset.')
    parser.add_argument('--seed',default=10)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--num_workers',type=int,default=2)

    parser.add_argument('--model_ckpt_AST', default='MIT/ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--save_best_ckpt',type=bool,default=False)
    parser.add_argument('--output_path',type=str,default='/checkpoints')
    parser.add_argument('--dataset_name',type=str,choices=['SPY'])
    parser.add_argument('--method',type=str,choices=['LoRA'])

    parser.add_argument('--seq_or_par', default = 'parallel', choices=['sequential','parallel'])
    parser.add_argument('--reduction_rate_adapter', type= int, default= 64)
    parser.add_argument('--adapter_type', type= str, default = 'Pfeiffer', choices = ['Houlsby', 'Pfeiffer'])
    parser.add_argument('--apply_residual', type= bool, default=False)
    parser.add_argument('--adapter_block', type= str, default='bottleneck', choices = ['bottleneck', 'convpass'])
    

    # LoRA params
    parser.add_argument('--reduction_rate_lora', type= int, default= 64)
    parser.add_argument('--alpha_lora', type= int, default= 8)

    parser.add_argument('--is_lora_ablation', type= bool, default= False)
    parser.add_argument('--lora_config', type = str, default = 'Wq,Wv', choices = ['Wq','Wq,Wk','Wq,Wv','Wq,Wk,Wv,Wo'])
    

    #Few shot experiments

    parser.add_argument('--is_few_shot_exp', default = False)
    parser.add_argument('--few_shot_samples', default = 64)

    return parser

def main(args):
    start_time = time.time()

    print("teste")
    print(args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    device = torch.device(args.device)

    if args.seed:

        seed = args.seed
        torch.manual_seed(seed)
        np.random.seed(int(seed))

    with open('ast_lora/hparams/train.yaml','r') as file:

        train_params = yaml.safe_load(file)

    
    if args.dataset_name == 'SPY':

        max_len_AST = train_params['max_len_AST_SPY']
        num_classes = train_params['num_classes_SPY']
        batch_size = train_params['batch_size_SPY']
        epochs = train_params['epochs_SPY']
    else:
        raise ValueError('The dataset you chose  is not supported as of now.')
    


    final_output = train_params['final_output']
    accuracy_folds = []
    # dataset = Spotify(data_path, max_len_AST, split, apply_SpecAug, few_shot, samples_per_class)

    if args.dataset_name == 'SPY':

        train_data = Spotify(args.data_train,max_len_AST,'train',apply_SpecAug=True,few_shot=args.is_few_shot_exp,samples_per_class=args.few_shot_samples)
        val_data =Spotify(args.data_val,max_len_AST,'valid',apply_SpecAug=False,few_shot=args.is_few_shot_exp,samples_per_class=args.few_shot_samples)
        test_data =Spotify(args.data_test,max_len_AST,'test',apply_SpecAug=False,few_shot=args.is_few_shot_exp,samples_per_class=args.few_shot_samples)
        print("Aqui")
        for i in range(len(train_data)):
            audio, label = train_data[i]
            print(f"Exe {i}: shape: {audio.shape}, label: {label}")
    
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=False)
    val_loader =DataLoader(val_data,batch_size=batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False,)
    test_loader = DataLoader(test_data,batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False)


    print(args.device)
    print("AQUI!")

    method = args.method

    if method == 'LoRA':
        model = AST_LoRA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
        lr = train_params['lr_LoRA']

    n_parameters = sum(p.numel() for p in model.parameters())
    print('Number of params of the model:', n_parameters)

    print(model)

    optimizer = AdamW(model.parameters(), lr= lr ,betas= (0.9,0.98),eps= 1e-6, weight_decay= train_params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*(epochs))

    print(f"Start training for {epochs} epochs")
        
    best_acc = 0.

    for epoch in range(epochs):

        train_loss,train_acc = train_one_epoch(model,train_loader,optimizer,scheduler,device,criterion)
        print(f"TrainLoss at epoch{epoch}:{train_loss}")


        val_loss,val_acc = eval_one_epoch(model,val_loader,device,criterion)

        if val_acc > best_acc:

            best_acc = val_acc
            best_params = model.state_dict()

            if args.save_best_ckpt:
                torch.save(best_params, os.getcwd()+args.output_path + f'/bestmodel')



        print("Train intent accuracy:",train_acc*100)
        print("Valid intent accuracy: ", val_acc*100)  
        current_lr = optimizer.param_groups[0]['lr']
        print('Learning rate after initialization:',current_lr)

    best_model = copy.copy(model)
    best_model.load_state_dict(best_params)
        
    test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)
      
    accuracy_folds.append(test_acc)

    print("Accuracy:", accuracy_folds)
    print("Std accuracy:",np.std(accuracy_folds))

    total_time = time.time() - start_time()
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))




if __name__=="__main__":
    parser = argparse.ArgumentParser('Lora Transfer-Learning of AST',
                                    parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

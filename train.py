import argparse
import os
from math import log10,exp,sqrt
import numpy as np
import scipy.spatial
from torch import nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
import torchvision.utils as utils
from torch.autograd import Variable, grad as torch_grad
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from torch import autograd

from data_utils import *
from loss import *
from model import Generator64, ML64


parser = argparse.ArgumentParser(description='Train Image Generation Models')
parser.add_argument('--data_path', default='/data/mendai/celeba', type=str, help='dataset path')
parser.add_argument('--name', default='results', type=str, help='path to store results')
parser.add_argument('--size', default=64, type=int, help='training images size')
parser.add_argument('--out_dim', default=10, type=int, help='ML network output dim')
parser.add_argument('--num_epochs', default=80, type=int, help='train epoch number')
parser.add_argument('--num_samples', default=64, type=int, help='number of displayed samples')
parser.add_argument('--batch_size', default=64, type=int, help='train batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='train learning rate')
parser.add_argument('--beta1', default=0, type=float, help='Adam optimizer beta1')
parser.add_argument('--beta2', default=0.9, type=float, help='Adam optimizer beta2')
parser.add_argument('--load_model', default = 'no', type=str, choices=['yes', 'no'], help='if load previously trained model')
parser.add_argument('--g_model_name', default='', type=str, help='generator model name')
parser.add_argument('--ml_model_name', default='', type=str, help='metric learning model name')
parser.add_argument('--margin', default=1, type=float, help='triplet loss margin')
parser.add_argument('--alpha', default=1e-2, type=float, help='triplet loss direction guidance weight parameter')
parser.add_argument('--n_threads', type=int, default=8)



if __name__ == '__main__':
    opt = parser.parse_args()
    
    SIZE = opt.size
    out_dim = opt.out_dim
    learning_rate = opt.lr
    beta1 = opt.beta1
    beta2 = opt.beta2
    batch_size = opt.batch_size
    NUM_EPOCHS = opt.num_epochs
    num_samples = opt.num_samples
    LOAD_MODEL = opt.load_model
    G_MODEL_NAME = opt.g_model_name
    ML_MODEL_NAME = opt.ml_model_name
    margin = opt.margin
    alpha = opt.alpha
    
    
    output_path = opt.name 
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sample_path = output_path           

    trainset = TrainDatasetFromFolder(opt.data_path, size=SIZE)  
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    img_size = (SIZE, SIZE, 3) 
       
    # Generator
    netG = Generator64() .to(device)  
    optimizerG = optim.Adam(netG.parameters(), lr = learning_rate, betas=(beta1,beta2),eps= 1e-6) 
    
    # ML
    netML = ML64(out_dim=out_dim).to(device)      
    optimizerML = optim.Adam(netML.parameters(), lr = learning_rate, betas=(0.5,0.999),eps= 1e-3)   

    # Losses    
    triplet_ = TripletLoss(margin, alpha).to(device)    


    if LOAD_MODEL == 'yes':
        netG.load_state_dict(remove_module_str_in_state_dict(torch.load(str(output_path + "/generator_latest.pt"))))
        netML.load_state_dict(remove_module_str_in_state_dict(torch.load(str(output_path + "/ml_model_latest.pt")))) 
        print("LOAD MODEL SUCCESSFULLY")
        
    
    #fixed_noise1 = gen_rand_noise(1).to(device) 

    for epoch in range(1, NUM_EPOCHS + 1):
            
        train_bar = tqdm(train_loader) 
    
        netG.train()
        netML.train()        
        
        for target in train_bar:                                                    
                                                     
            real_img = Variable(target).cuda()           
                
            batch_size = real_img.size()[0]
            if batch_size != opt.batch_size:
                continue
                
            z = torch.randn(batch_size, 128, 1, 1).to(device)
            z.requires_grad_(True)    
            fake_img = netG(z)
            
#             print(real_img.size())
#             print(fake_img.size())
            
            ############################ 
            # Metric Learning
            ############################
            
            netML.zero_grad()   
            fake_img = netG(z) 
            ml_real_out, _ = netML(real_img)
            ml_fake_out, _ = netML(fake_img)
            
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])                

            ############################
            
            pd_r = pairwise_distances(ml_real_out, ml_real_out) 
            pd_f = pairwise_distances(ml_fake_out, ml_fake_out)
        
            p_dist =  torch.dist(pd_r,pd_f,2)   
            c_dist = torch.dist(ml_real_out.mean(0),ml_fake_out.mean(0),2)      

            g_loss = p_dist + c_dist 
            netG.zero_grad()
            g_loss.backward()                        
            optimizerG.step()
            
            fake_img = netG(z) 
            ml_real_out = netML(real_img)
            ml_fake_out = netML(fake_img.detach())           
            
            r1=torch.randperm(batch_size)
            r2=torch.randperm(batch_size)
            ml_real_out_shuffle = ml_real_out[r1[:, None]].view(ml_real_out.shape[0],ml_real_out.shape[-1])
            ml_fake_out_shuffle = ml_fake_out[r2[:, None]].view(ml_fake_out.shape[0],ml_fake_out.shape[-1])

            ml_loss = triplet_(ml_real_out,ml_real_out_shuffle,ml_fake_out_shuffle) 
            ml_loss.backward()
            optimizerML.step()  


            train_bar.set_description(desc='[%d/%d]' % (epoch, NUM_EPOCHS))

        if epoch % 5 != 0:
            continue    
        
###--------------------Hausdorff distance----------------------------------------
        h_distance, idx1, idx2 = scipy.spatial.distance.directed_hausdorff(ml_real_out.clone().cpu().detach().numpy(), ml_fake_out.clone().cpu().detach().numpy(), seed=0)
        print('hausdorff distance: ', h_distance)        
            
        print(' c_dist:',c_dist.item(), ' p_dist:', p_dist.item(),' triplet_loss:',triplet_loss.item())

###------------------display generated samples--------------------------------------------------
        fixed_noise = gen_rand_noise(num_samples).to(device)        
        gen_images = generate_image(netG, dim=SIZE, batch_size=num_samples, noise=fixed_noise)
        utils.save_image(gen_images, str(sample_path  +'/' + 'samples_{}.png').format(epoch), nrow=int(sqrt(num_samples)), padding=2)             
        
# 	#----------------------Save model----------------------
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_{}.pt").format(epoch))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_{}.pt").format(epoch))
        torch.save(netG.state_dict(), str(output_path  +'/' + "generator_latest.pt"))
        torch.save(netML.state_dict(), str(output_path  +'/' + "ml_model_latest.pt"))
       

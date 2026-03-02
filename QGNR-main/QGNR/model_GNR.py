import time
import ot
from ot.gromov import gromov_wasserstein2
import numpy as np
from helper import *
from siren_pytorch import *
from tqdm import tqdm

'''
Learning with SIREN using proximal gradient
'''
class GNR(nn.Module):
    '''
    - d_hidden: hidden units for the MLP of SIREN; [h1,h2] means 2 layers, h1 units in first layer, h2 units in second layer
    - w0: frequency for SINE activation
    '''

    def __init__(self, d_hidden=[20,20,20], w0=30., device='cpu', model='QGNR',
                 q_hidden=4, q_spectrum_layer=6):
        super(GNR, self).__init__()

        if model == "IGNR":
            self.net = SirenNet(
                                dim_in = 2,
                                dim_hidden = d_hidden,
                                dim_out = 1,
                                num_layers = len(d_hidden),
                                final_activation = 'sigmoid',
                                w0_initial = w0).to(device)
        else:
            self.net = Hybridren(in_features=2, out_features=1,
                                 hidden_features=q_hidden,
                                 spectrum_layer=q_spectrum_layer).to(device)

        self.device = device
    def sample(self,M,f_sample='fixed'):
        if f_sample=='fixed':
            x = (torch.arange(M)+(1/2))/M
            y = (torch.arange(M)+(1/2))/M
        else:
            x = torch.sort(torch.rand(M))[0]
            y = x

        xx,yy = torch.meshgrid(x,y, indexing='ij')

        mgrid=torch.stack([xx,yy],dim=-1)
        mgrid=rearrange(mgrid, 'h w c -> (h w) c')
        mgrid = mgrid.to(self.device)
        C_recon_tmp = self.net(mgrid)

        C_recon_tmp = torch.squeeze(rearrange(C_recon_tmp, '(h w) c -> h w c', h = M, w = M))

        # when training only half plane
        C_recon_tmp = torch.triu(C_recon_tmp,diagonal=1)
        C_recon_tmp = C_recon_tmp+torch.transpose(C_recon_tmp,0,1)


        return C_recon_tmp

    def fun_loss_pg(self,M,h_recon,C_input,h_input,f_sample='fixed',G0_prior=None,G0_cost=None):
        # loss that compute gw2 using proximal gradient from Xu et al 2021
        C_recon_tmp = self.sample(M,f_sample=f_sample)
        loss,T = gwloss_pg(C_recon_tmp,C_input,h_recon,h_input,G0_prior=G0_prior,G0_cost=G0_cost)

        return loss,T

    def fun_loss_cg(self,M,h_recon,C_input,h_input,f_sample='fixed'):
        # loss that compute gw2 using the default conditional gradient method 
        C_recon_tmp = self.sample(M,f_sample=f_sample)
        loss,log = gromov_wasserstein2(C_recon_tmp,C_input,h_recon,h_input,log=True)
        return loss,log['T']



class GNR_wrapper:
    '''
    Traing IGNR
    '''

    def __init__(self, d_hidden=[20,20,20], w0=30., model='QGNR',
                 q_hidden=4, q_spectrum_layer=6):
        self.mlp = GNR(d_hidden, w0, model=model,
                       q_hidden=q_hidden, q_spectrum_layer=q_spectrum_layer)
        print('parameters:%d' % (sum(p.numel() for p in self.mlp.parameters() if p.requires_grad)))

    # def train(self, graphs, K='input', n_epoch=80,lr=0.1,f_sample='fixed'):
    #     optim = torch.optim.Adam([*self.mlp.parameters()],lr=lr)

    #     M = len(graphs)
    #     loss_l=[]
    #     trans = [None]*M

    #     # --- 修改点 1：使用 tqdm 包装 range(n_epoch) ---
    #     epoch_pbar = tqdm(range(n_epoch), desc="Training Epochs", leave=False)
        
    #     for epoch in epoch_pbar:
    #     # ----------------------------------------------
    #         loss = []

    #         for i in range(M):
    #             num_node_i = graphs[i].shape[0]
    #             h_input = torch.from_numpy(ot.unif(num_node_i)).float()
    #             g_input = torch.from_numpy(graphs[i]).float().to(self.mlp.device)

    #             if K == 'input':
    #                 # reconstruct each same size as input
    #                 h_recon = torch.from_numpy(ot.unif(num_node_i)).float()
    #                 K_recon = num_node_i
    #             else:
    #                 # reconstruct to a specified size K
    #                 h_recon = torch.from_numpy(ot.unif(K)).float()
    #                 K_recon = K

    #             loss_i,T_i = self.mlp.fun_loss_pg(K_recon,h_recon,g_input,h_input,f_sample=f_sample,G0_prior=None,G0_cost=trans[i])
    #             loss.append(loss_i)
    #             trans[i]=T_i

    #         loss = torch.stack(loss)
    #         loss = torch.mean(loss)
    #         loss.backward()
    #         optim.step()
    #         optim.zero_grad()
            
    #         # 提取当前的 loss 值
    #         current_loss = loss.item()
    #         loss_l.append(current_loss)
            
    #         # --- 修改点 2：在进度条后缀中实时更新 Loss ---
    #         epoch_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
    #         # ----------------------------------------------

    #     return loss_l
    def train(self, graphs, K='input', n_epoch=80, lr=0.1, f_sample='fixed'):
        optim = torch.optim.Adam([*self.mlp.parameters()], lr=lr)

        M = len(graphs)
        loss_l = []
        trans = [None] * M

        # 外层进度条：监控整体 Epoch 进度
        epoch_pbar = tqdm(range(n_epoch), desc="Training Epochs", leave=False)

        for epoch in epoch_pbar:
            loss = []

            # 内层进度条：监控当前 Epoch 下每张图的计算进度
            # 帮助排查"进度为0"是因为计算太慢还是死锁
            graph_pbar = tqdm(range(M), desc=f"Epoch {epoch} Graphs", leave=False)
            
            for i in graph_pbar:
                num_node_i = graphs[i].shape[0]
                h_input = torch.from_numpy(ot.unif(num_node_i)).float()
                g_input = torch.from_numpy(graphs[i]).float().to(self.mlp.device)

                if K == 'input':
                    # reconstruct each same size as input
                    h_recon = torch.from_numpy(ot.unif(num_node_i)).float()
                    K_recon = num_node_i
                else:
                    # reconstruct to a specified size K
                    h_recon = torch.from_numpy(ot.unif(K)).float()
                    K_recon = K

                # 核心耗时步骤：计算 GW 距离
                loss_i, T_i = self.mlp.fun_loss_pg(K_recon, h_recon, g_input, h_input, f_sample=f_sample, G0_prior=None, G0_cost=trans[i])
                loss.append(loss_i)
                trans[i] = T_i

            # 将列表转换为张量并计算均值
            loss_tensor = torch.stack(loss)
            mean_loss = torch.mean(loss_tensor)
            
            # 反向传播与优化
            mean_loss.backward()
            optim.step()
            optim.zero_grad()
            
            # 记录并显示当前 Loss
            current_loss = mean_loss.item()
            loss_l.append(current_loss)
            
            # 在外层进度条末尾实时更新 Loss 值
            epoch_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

        return loss_l
    def get_W(self,K):
        W = self.mlp.sample(K)
        return W.detach().cpu().numpy()





        





from model_GNR import *
from helper import *
import pickle
import argparse
import os



"""
Adapted from:
Xia, X., Mishne, G., & Wang, Y., "Implicit Graphon Neural Representation",
In International Conference on Artificial Intelligence and Statistics,
10619–10634 (PMLR, 2023).

This code implements a quantum graphon learning version of the above method.
"""



parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='Data/graphs.pkl',
                    help='path to graphs.pkl')
parser.add_argument('--n-epoch', type=int, default=80,
                    help='number of traning epochs')
parser.add_argument('--f-sample', type=str, default='fixed',
                    help='grid sampling strategy')
parser.add_argument('--w0', type=float, default=30,
                    help='default frequency for sine activation')
parser.add_argument('--mlp_dim_hidden', dest='mlp_dim_hidden', type=str, default="20,20,20",
					help='hidden units per layer for SIREN')
parser.add_argument('--model', type=str, default="QGNR",
					help='use QGNR or IGNR')
parser.add_argument('--f-name', type=str, default='res',
					help='saving directory name')
args = parser.parse_args()



ppath = os.path.dirname(os.path.abspath(__file__))
with open(args.data_path, 'rb') as f:
 data=pickle.load(f)

n_trials = len(data[0])
n_funs   = len(data)

error = np.zeros((n_funs,n_trials)) # gw error

time_mat = np.zeros((n_funs,n_trials))
loss_mat = np.zeros((n_funs,args.n_epoch))


# define savepath
save_path = ppath+'/Result/'+args.f_name+'/'
print(save_path)
if not os.path.exists(save_path):
  os.makedirs(save_path)
print('saving path is:'+save_path)


exp_inds = range(n_funs)  # derived from data, not hardcoded
args.mlp_dim_hidden = [int(x) for x in args.mlp_dim_hidden.split(',')]

for i_exp in exp_inds:
	graphon0 = synthesize_graphon(r=1000, type_idx=i_exp) #ground-truth graphon sampled at resolution 1000
	np.fill_diagonal(graphon0,0.) #ignore diagonal entries when computing GW error

	for i_trial in range(n_trials):

		graphs = data[i_exp][i_trial]
		all_graphs = graphs



		gl_mlp = GNR_wrapper(args.mlp_dim_hidden, w0=args.w0, model=args.model)
		loss = gl_mlp.train(all_graphs, K='input', n_epoch=args.n_epoch, f_sample=args.f_sample)
		loss_mat[i_exp] = loss  # store loss curve for this experiment
		with torch.no_grad():

			W1 = gl_mlp.get_W(1000) #get estimated graphon at resolution 1000
			tmp_gw = gw_distance(graphon0,W1)

			error[i_exp,i_trial]=tmp_gw
			print('Data {}\tTrial {}\tError={:.3f}\t'.format(
					i_exp, i_trial, error[i_exp,i_trial]))


	np.set_printoptions(suppress=True)
	print('miu:',np.mean(error[i_exp, :]))
	print('std:',np.std(error[i_exp, :]))



np.set_printoptions(suppress=True)
print(np.round(np.mean(error[exp_inds,:], axis=1),3))
print(np.round(np.std(error[exp_inds,:], axis=1),3))











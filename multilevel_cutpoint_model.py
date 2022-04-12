from scipy.optimize import minimize
import scipy.stats
import torch
import torch.optim as optim
import torch.nn as nn
import scipy
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from node import Node
# from copy import deepcopy

# torch.autograd.set_detect_anomaly(True)
class CutpointModel(nn.Module):
    def __init__(self,
                 X_train, t_train, s_train,
                 sigmoid_temperature = 0.01, depth = 1, iterations = 1000,
                prior_strength = 1):
        
        super(CutpointModel, self).__init__()
      
        self.X_train = torch.from_numpy(X_train).float()
        self.t_train = torch.from_numpy(t_train).float()
        self.s_train = torch.from_numpy(s_train).float()
        self.prior_strength = prior_strength
        
        self.sigmoid_temperature = sigmoid_temperature 
        self.depth = depth
        self.iterations = iterations
                
        p = self.X_train.shape[1]
        
        # set up parameters
        self.cutpoint0 = self.cutpoint_init(self.depth)
        self.cutpoints = nn.ParameterList([ nn.Parameter(torch.tensor(cutpoint.item())) for cutpoint in self.cutpoint0])
                
        self.layers = []
        self.layer_params = nn.ParameterList()
        
        for ii in range(depth+1):
    
            layer1 = torch.nn.Linear(p, 2**(ii + 1) ) # where b is the number of buckets at that level

            torch.nn.init.xavier_uniform_(layer1.weight)

    
            self.layers.append(
                torch.nn.Sequential(
                    layer1,
                    torch.nn.Softmax(dim=1)
                )
            )
        
            self.layer_params.extend(
                self.layers[ii].parameters()
            )
    
        # set up cutpoint tree
        self.root = Node(0)
        self.root.left_bound = min(self.t_train)
        self.root.right_bound = max(self.t_train)
        self.root = Node.build_tree(self.root, depth)
        
     
    
    # evenly spaced cutpoints
    def cutpoint_init(self, depth):
        root = Node(0)
        Node.build_tree(root, depth)
        inorder_traversal = Node.inorder(root, depth)
        equal_spacings = [(i+1)/(2**(depth+1)) for i in range(0, 2**(depth+1) + -1)]
        cutpoint0 = [0] * len(inorder_traversal)

        for i in range(0, len(inorder_traversal)):
            index = inorder_traversal[i].index
            cutpoint0[index] = equal_spacings[i]/2
#             cutpoint0[index] = equal_spacings[i]


        return torch.tensor(cutpoint0)
    
    def logit(self, x):
        return torch.log(x/ (1 - x))    
    
    def multinomial_loss(self, X, t, s, layer_net, cutpoints):
        
        t_scaled = (t - min(t)) / (max(t) - min(t))

        likelihood = 0
        prior = 0
        
        # example
        # f_t_i = [0.1, 0.3, 0.2, 0.4]
        # F_t_i =  [1.0, 0.9, 0.6, 0.4]
        
        f_t = layer_net(X)

        F_t = f_t +  torch.sum(f_t, dim=1, keepdims=True) - torch.cumsum(f_t, dim=1)


        left_boundary = torch.transpose(torch.stack([torch.sigmoid((t_scaled-lb)/ self.sigmoid_temperature) for lb in [0] + cutpoints]),0 ,1)
        right_boundary = torch.transpose(torch.stack([1-torch.sigmoid((t_scaled-rb)/ self.sigmoid_temperature) for rb in cutpoints + [1]]), 0, 1)
        

#         parta = torch.unsqueeze(s,1) * f_t
#         partb = (1 - torch.unsqueeze(s,1)) * F_t
#         nll = -1 * torch.mean(torch.sum(torch.log((parta + partb) * left_boundary * right_boundary), axis=1))

        ll_event = torch.log(torch.sum(f_t * left_boundary * right_boundary, axis=1))
        ll_censored = torch.log(torch.sum(F_t * left_boundary * right_boundary, axis=1))
        ll = torch.unsqueeze(s,1) * ll_event + (1 - torch.unsqueeze(s,1)) * ll_censored

        nll = -1 * torch.mean(ll)
        
        # prior loop
        for lb, curr, rb in zip([0] + cutpoints[:-1], cutpoints, cutpoints[1:] + [1]):
            
            temp = (curr - lb) / (rb - lb)
            prior += -1 * self.prior_strength * torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.5)).log_prob(temp)
            
        return nll + prior
    
    
    def train(self):
        flat_layers = [item for sublist in self.layers for item in sublist]
                    
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        
        loss = 0
        loss_layers = [0 for i in range(0, self.depth+1)]
        iteration_num = 1

        ######################################
        # training iteration

        while iteration_num < self.iterations:

#             if iteration_num % 10 == 0:
                
            #####################################
            # inorder traversal
            for current_depth in range(0, self.depth+1):

                nodes = Node.inorder(self.root, current_depth)
                indices = [node.index for node in nodes]
                current_layers = self.layers[current_depth]
                current_cutpoints = [self.cutpoints[i] for i in indices]
                loss_layers[current_depth] = self.multinomial_loss(self.X_train, self.t_train, self.s_train,
                                                                   current_layers, current_cutpoints)
            
            # end inorder traversal
            ###################################
            loss = sum(loss_layers)
            
            print(iteration_num)
            print(f"loss {loss}")
            print(f"cutpoint {self.cutpoints[0].item()}")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # one optimization step
            iteration_num += 1
            
        # end training iteration
        ######################################

        

    #######################################
    def plot_result(self, t_true, t_false, title = "Survival distribution", label_true="True", label_false="False"):
        
        max_t = max(max(t_true), max(t_false))
        min_t = min(min(t_true), min(t_false))
        
        cutpoints = [ cutpoint * (max_t - min_t) + min_t for cutpoint in self.cutpoints ]
        cutpoints_init = [cutpoint * (max_t-min_t) + min_t for cutpoint in self.cutpoint0]

        plt.hist(t_true, color="red", alpha = 0.5, label=label_true)
        plt.hist(t_false, color="blue", alpha = 0.5, label=label_false)

        plt.legend()
        plt.title(title + ": cutpoints init")

        for cutpoint in cutpoints_init:
            plt.axvline(cutpoint)
        plt.show()

        plt.hist(t_true, color="red", alpha = 0.5, label=label_true)
        plt.hist(t_false, color="blue", alpha = 0.5, label=label_false)

        plt.legend()
        plt.title( title + ": cutpoints learned")

        for cutpoint in cutpoints:
            plt.axvline(cutpoint)
        plt.show()
        
        
#     def predict():
        

        
        
# def discrete_ci(cutpoints, model, X_test_in, t_test_in, s_test_in, t_train_in):

#     s_test = torch.from_numpy(s_test_in).float()
#     X_test = torch.from_numpy(X_test_in).float()
#     t_test = torch.from_numpy(t_test_in).float()
#     t_train = torch.from_numpy(t_train_in).float()

#     # predicted bucket probabilities for test data
#     pred = model(X_test).detach().numpy()
    
#     # cdfs
#     t_pred_cdf = np.cumsum(pred, axis=1) 

#     bucket_boundaries = [0] + sorted([i.item() for i in cutpoints]) + [1]
#     # rescale
#     bucket_boundaries = [boundary_i * (max(t_train) - min(t_train)) + min(t_train) 
#                              for boundary_i in bucket_boundaries]
    
#     N = len(pred)
#     n_buckets = len(bucket_boundaries) - 1

#     # one hot vector for where the 1 is in the most likely bucket
#     t_true_idx = np.zeros((N, n_buckets),dtype=int)
#     for ii in range(N):
#         for jj in range(n_buckets):
#             if t_test[ii] < bucket_boundaries[jj+1]:
#                 t_true_idx[ii][jj] = 1
#                 break
    
#     t_true_idx = np.argmax(t_true_idx, axis=1)
#     concordant = 0
#     total = 0

#     idx = np.arange(N)

    
    
#     for i in range(N):

#         if s_test[i] == 0:
#             continue

#         # time bucket of observation for i, then for all but i
#         tti_idx = t_true_idx[i]
        
#         tt_idx = t_true_idx[idx != i]

#         # calculate predicted risk for i at the time of their event
#         tpi = t_pred_cdf[i, tti_idx]


#         # predicted risk at that time for all but i
#         tp = t_pred_cdf[idx != tti_idx, tti_idx]

        
#         total += np.sum(tti_idx < tt_idx) # observed in i first

#         concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

#     return concordant / total

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
                 sigmoid_temperature = 0.01, 
                 depth = 1, 
                 iterations = 1000,
                 prior_bool = True,
                 prior_strength = 1,
                 hidden_size = 32,
                 init_method="even"):
        
        super(CutpointModel, self).__init__()
      
        self.X_train = torch.from_numpy(X_train).float()
        self.t_train = torch.from_numpy(t_train).float()
        self.s_train = torch.from_numpy(s_train).float()
        
        # if True: will add prior to loss
        # if False: will add entropy to loss
        self.prior_bool = prior_bool
        # multiplier on prior/entropy
        self.prior_strength = prior_strength
        
        self.sigmoid_temperature = sigmoid_temperature 
        self.depth = depth
        self.iterations = iterations
                        
        p = self.X_train.shape[1]
        
        # set up parameters
        self.cutpoint0 = self.cutpoint_init_even(self.depth) if init_method=="even" else self.cutpoint_init_quantile(self.depth, self.t_train)
        self.cutpoints = nn.ParameterList([ nn.Parameter(torch.tensor(cutpoint.item())) for cutpoint in self.cutpoint0])
                
        self.layers = []
        self.layer_params = nn.ParameterList()
        
        for ii in range(depth+1):
            # number of buckets at that level
            b = 2**(ii + 1)
            layer1 = torch.nn.Linear(p, hidden_size)
            layer2 = torch.nn.Linear(hidden_size, b)

#             torch.nn.init.xavier_uniform_(layer1.weight)
    
            self.layers.append(
                torch.nn.Sequential(
                    layer1,
                    torch.nn.ReLU(),
                    layer2,
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
    def cutpoint_init_even(self, depth):
        root = Node(0)
        Node.build_tree(root, depth)
        inorder_traversal = Node.inorder(root, depth)
        equal_spacings = [(i+1)/(2**(depth+1)) for i in range(0, 2**(depth+1) + -1)]
        cutpoint0 = [0] * len(inorder_traversal)

        for i in range(0, len(inorder_traversal)):
            index = inorder_traversal[i].index
#             cutpoint0[index] = equal_spacings[i]/2
            cutpoint0[index] = equal_spacings[i]


        return torch.tensor(cutpoint0)
    
    
    def cutpoint_init_quantile(self, depth, t):
        root = Node(0)
        Node.build_tree(root, depth)
        inorder_traversal = Node.inorder(root, depth)
        t_conv = (t - min(t)) / (max(t) - min(t))
        
        quantile_spacings = np.linspace(0, 1, 2**(depth+1)+1)
        quantile_spacings = np.quantile(t_conv, quantile_spacings)[1:-1]

        cutpoint0 = [0] * len(inorder_traversal)

        for i in range(0, len(inorder_traversal)):
            index = inorder_traversal[i].index
            cutpoint0[index] = quantile_spacings[i]


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
        
        approx_p = torch.mean(left_boundary * right_boundary, axis=0)
        
        # if prior
        if self.prior_bool:
             # prior loop
            for lb, curr, rb in zip([0] + cutpoints[:-1], cutpoints, cutpoints[1:] + [1]):
                temp = (curr - lb) / (rb - lb)
                prior += -1 * self.prior_strength * torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.5)).log_prob(temp)
                nll += prior
        # else use entropy
        else:
            entropy = -1 * self.prior_strength * torch.sum(approx_p * torch.log(approx_p))
            nll -= entropy

        return nll

    
    
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
#             print(f"cutpoint {self.cutpoints[0].item()}")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # one optimization step
            iteration_num += 1
            
        # end training iteration
        ######################################

        

    #######################################
    def plot_result(self, t_true, title = "Survival distribution", label_true="endpoint"):
        
        max_t = max(self.t_train)
        min_t = min(self.t_train)
        
        cutpoints = [ cutpoint * (max_t - min_t) + min_t for cutpoint in self.cutpoints ]
        cutpoints_init = [cutpoint * (max_t-min_t) + min_t for cutpoint in self.cutpoint0]

        plt.hist(t_true, color="red", bins=50, alpha = 0.5, label=label_true)
#         plt.hist(t_false, color="blue",bins=50,  alpha = 0.5, label=label_false)
        plt.xlabel('Survival time')
        plt.legend()
        plt.title(title + ": cutpoints init")

        for cutpoint in cutpoints_init:
            plt.axvline(cutpoint)
        plt.show()

        plt.hist(t_true, color="red", bins=50, alpha = 0.5, label=label_true)
#         plt.hist(t_false, color="blue", bins=50, alpha = 0.5, label=label_false)
        plt.xlabel('Survival time')
        plt.legend()
        plt.title( title + ": cutpoints learned")

        for cutpoint in cutpoints:
            plt.axvline(cutpoint)
        plt.show()
        
        
    def predict(self, X_test):
        return self.layers[-1](torch.from_numpy(X_test).float()).detach().numpy()
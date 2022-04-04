from scipy.optimize import minimize
import scipy.stats
import torch
import torch.optim as optim
import scipy
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from Node import Node

# torch.autograd.set_detect_anomaly(True)
class Model:
    def __init__(self, X, t, sigmoid_temp = 0.01, depth = 1, cutpoint0=None, iterations = 1000):
        self.X = torch.tensor(X, dtype=torch.float)
        self.t = torch.tensor(t, dtype=torch.float)
        self.t_convert = (self.t - min(self.t)) / (max(self.t) - min(self.t))
        self.sigmoid_temp = sigmoid_temp 
        self.depth = depth
        self.iterations = iterations
        
        self.n_cutpoints = 2**(depth+1)-1
        self.theta = []
        for curr_depth in range(0, self.depth+1):
            self.theta.append(
                [torch.zeros( X.shape[1]+1, requires_grad=True).float()]
                * (sum([2**(i) for i in range(0, curr_depth+1)])+1))
#             self.theta.append([torch.zeros( X.shape[1]+1, requires_grad=True).float() for i in range(0,2^(curr_depth)+1)])

#         self.theta = [torch.zeros( X.shape[1]+1, requires_grad=True).float() for i in range(0,self.n_cutpoints)]
        
        if (cutpoint0 is None):
            self.cutpoint0 = self.cutpoint_init(self.depth)
        else:
            self.cutpoint0 = cutpoint0
    
        self.cutpoint_logit = [torch.tensor(self.logit(torch.tensor(cutpoint)).item(), requires_grad=True) for cutpoint in self.cutpoint0]
    
        # set up cutpoint tree
        self.root = Node(0)
        self.root.left_bound = min(self.t)
        self.root.right_bound = max(self.t)
        self.root = Node.build_tree(self.root, depth)
        
      
    def cutpoint_init(self, depth):
        root = Node(0)
        Node.build_tree(root, depth)
        inorder_traversal = Node.inorder(root, depth)
        equal_spacings = [(i+1)/(2**(depth+1)) for i in range(0, 2**(depth+1) + -1)]
        cutpoint0 = [0] * len(inorder_traversal)

        for i in range(0, len(inorder_traversal)):
            index = inorder_traversal[i].index
            cutpoint0[index] = equal_spacings[i]/2

        return cutpoint0
    
    def logit(self, x):
        return torch.log(x/ (1 - x))
    
    # shifted and scaled sigmoid function
    def sigmoid(self, x, a=0, b=1.):
        return 1 / (1 + (torch.exp(-1 * (x - a) / b)))
    
    
    def pred_value(self, x, theta):
        prod = torch.matmul(x,theta)
        return self.sigmoid(prod)
    
    
    def multinomial_loss(self, thetas, cutpoint_logits):

        
        t_scaled = (self.t - min(self.t)) / (max(self.t) - min(self.t))
        x1 = torch.cat([self.X, torch.ones((len(self.X), 1))], axis=1)

        cutpoints = [self.sigmoid(cp_logit) for cp_logit in cutpoint_logits]
        likelihood = 0
        prior = 0
        
        for lb, rb, theta in zip([0] + cutpoints, cutpoints + [1], thetas):
            left_boundary  = self.sigmoid(t_scaled, lb, self.sigmoid_temp)
            right_boundary = self.sigmoid(t_scaled, rb, self.sigmoid_temp)
        
            p_hat = self.pred_value(x1, theta)
            likelihood += -1 * torch.mean(torch.log(p_hat) * left_boundary * right_boundary)
    
        
        # prior loop
        for lb, curr, rb in zip([0] + cutpoints[:-1], cutpoints, cutpoints[1:] + [1]):
#             print(f"lb:\t{lb}\tcurr:\t{curr}\trb:\t{rb}")
            
            temp = (curr - lb) / (rb - lb)
            prior += -1 * torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.5)).log_prob(temp)
            
#         print('Likelihood = %.7e | Prior = %.7e' % (likelihood, prior))

        return likelihood + prior
    
    
    def train(self):
        flat_theta = [item for sublist in self.theta for item in sublist]

        optimizer = optim.Adam(flat_theta + self.cutpoint_logit, lr=0.001)
        loss = 0
        loss_layers = [0 for i in range(0, self.depth+1)]
        iteration_num = 1

        ######################################
        # training iteration

        while iteration_num < self.iterations:

#             if iteration_num % 10 == 0:
#                 print(iteration_num)
                
            #####################################
            # inorder traversal
            for current_depth in range(0, self.depth+1):

                nodes = Node.inorder(self.root, current_depth)
                indices = [node.index for node in nodes]
                current_thetas = self.theta[current_depth]
                current_cutpoints_logit = [self.cutpoint_logit[i] for i in indices]
                loss_layers[current_depth] = self.multinomial_loss(current_thetas, current_cutpoints_logit)
            
            # end inorder traversal
            ###################################
            loss = sum(loss_layers)
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
        
        cutpoints = [ self.sigmoid(cutpoint) * (max_t - min_t) + min_t for cutpoint in self.cutpoint_logit ]
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


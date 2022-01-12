import torch
import torch.optim as optim
import scipy
# torch.autograd.set_detect_anomaly(True)


class Model:
    def __init__(self, X, t, theta=None, cutpoint=None):
        self.X = torch.tensor(X, dtype=torch.float)
        self.t = torch.tensor(t, dtype=torch.float)
                
        if theta is None:
            self.theta = torch.zeros(X.shape[1]+1, requires_grad=True).float()
        else:
            self.theta = theta
            
        if cutpoint is None:
            self.cutpoint_logit = torch.tensor([-1.2], requires_grad=True)
        else:
            cutpoint_tensor = torch.tensor([cutpoint])
            self.cutpoint_logit = torch.tensor([self.logit(cutpoint_tensor).item()], requires_grad=True)
            
        self.cutpoint = self.sigmoid(self.cutpoint_logit)

    def logit(self, x):
        return torch.log(x/ (1 - x))
    
    # shifted and scaled sigmoid function
    def sigmoid(self, x, a=0, b=1.):
        return 1 / (1 + (torch.exp(-1 * (x - a) / b)))
    
    
    def pred_value(self, x, theta):
        prod = torch.matmul(x,theta)
        return self.sigmoid(prod)
    
    
    def loss_func(self, params, x, t):

        theta = params[:-1][0]

        cutpoint_logit = params[-1]

        cutpoint = self.sigmoid(self.cutpoint_logit)

        t_disc = self.sigmoid(t, cutpoint, 0.01)

        x1 = torch.cat([x, torch.ones((len(x), 1))], axis=1)

        p_hat = self.pred_value(x1, self.theta)

        n = len(t)
        likelihood = (-1/n) * torch.sum( t_disc * torch.log(p_hat) + (1-t_disc)*torch.log(1-p_hat) )

        prior = -1 * scipy.stats.beta.logpdf(cutpoint.detach().numpy(), 1.5, 1.5)[0]

#         print('Likelihood = %.7e | Prior = %.7e' % (likelihood, prior))

        return likelihood + prior
    
    
    def train(self):
        # then optimize
        # https://pytorch.org/docs/stable/optim.html
        optimizer = optim.Adam([self.theta, self.cutpoint_logit], lr=0.001)

        loss_diff = 1000
        loss_curr = 1000
        loss_prev = 0 

        iteration_num = 1

        tolerance = 1e-6

        while loss_diff > tolerance:

            iteration_num += 1
            loss_prev = loss_curr

            optimizer.zero_grad()
            loss_curr = loss = self.loss_func((self.theta, self.cutpoint_logit), self.X, self.t)
            loss.backward()
            optimizer.step()
            loss_diff = torch.norm(loss_curr - loss_prev) / (1 + torch.norm(loss_curr) )

        self.cutpoint = self.sigmoid(self.cutpoint_logit)
#         print(iteration_num)
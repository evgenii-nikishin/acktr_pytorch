import torch
from torch.autograd import Variable
from torch import optim

from math import sqrt

class KFAC_Optim(optim.Optimizer):
    def __init__(self, model, lr=0.01, delta=1e-3, update_num_iters=1, eps=1e-6):
        '''
            model: it is exactly a model we want to optimize. Not model.parameters()
            lr: learning rate
            delta: trush region parameter
            update_num_iters: recalculate fisher matrix every 'update_num_iters' steps
            eps: accuracy parameter
        '''
        super(KFAC_Optim, self).__init__(model.parameters(), dict())

        # save model
        self.model = model
        
        # create SGD optimizer
        self.lr = lr
        self.delta=delta
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.update_num_iters = update_num_iters
        self.iter_i = 0
        self.eps = eps
        
        # setting hooks to save input of layer and grad
        def forw_hook_lin(module, input):
            cur_input = input[0]
            if len(cur_input.shape) == 1:
                cur_input = cur_input.unsqueeze(0)
            
            module.__kfac_data__['aaT'] += torch.bmm(cur_input.unsqueeze(2), cur_input.unsqueeze(1)).sum(dim=0)
            module.__kfac_data__['act_num'] += cur_input.shape[0] 
    
    
        def back_hook_lin(module, grad_input, grad_output):
            cur_grad = grad_output[0]
            if len(cur_grad.shape) == 1:
                cur_grad = cur_grad.unsqueeze(0)
            
            module.__kfac_data__['dLdLT'] += torch.bmm(cur_grad.unsqueeze(2), cur_grad.unsqueeze(1)).sum(dim=0)
            module.__kfac_data__['grad_num'] += cur_grad.shape[0]
                    
        for name, module in model.named_modules():
            module_name = module.__class__.__name__
    
            if module_name == 'Linear':
                if not hasattr(module, "__kfac_data__"):
                    module.__kfac_data__ = dict()
                    module.__kfac_data__['act_num'] = 0
                    module.__kfac_data__['grad_num'] = 0
                    module.__kfac_data__['aaT'] = Variable(torch.zeros((module.weight.shape[1], module.weight.shape[1])))
                    module.__kfac_data__['dLdLT'] = Variable(torch.zeros((module.weight.shape[0], module.weight.shape[0])))

            
                module.register_forward_pre_hook(forw_hook_lin)
                module.register_backward_hook(back_hook_lin)

    def compute_fisher_grads(self, model):
        for name, module in model.named_modules():
            module_name = module.__class__.__name__
    
            if module_name in ['Linear']:
                # make update of Fisher matrix componets every update_num_iters iters
                if self.iter_i % self.update_num_iters == 0:            
                    aaT = module.__kfac_data__['aaT'] / module.__kfac_data__['act_num']
                    dLdLT = module.__kfac_data__['dLdLT'] / module.__kfac_data__['grad_num']
            
                    module.__kfac_data__['aaT'] = module.__kfac_data__['aaT'] / self.update_num_iters
                    module.__kfac_data__['dLdLT'] = module.__kfac_data__['dLdLT'] / self.update_num_iters
                    module.__kfac_data__['act_num']  = module.__kfac_data__['act_num'] // self.update_num_iters
                    module.__kfac_data__['grad_num'] = module.__kfac_data__['grad_num'] // self.update_num_iters
            
                    try:
                        if torch.norm(aaT).data.numpy()[0] < self.eps:
                            aaT = aaT + Variable(self.eps * torch.randn(aaT.shape))
                            
                        if torch.norm(dLdLT).data.numpy()[0] < self.eps:
                            dLdLT = dLdLT + Variable(self.eps * torch.randn(dLdLT.shape))
                           
                    
                        U1, S1, V1 = torch.svd(aaT)
                        U2, S2, V2 = torch.svd(dLdLT)
                
                        inv_S1 = torch.zeros_like(S1)
                        inv_S1[S1 > self.eps] = 1 / S1[S1 > self.eps]
                
                        inv_S2 = torch.zeros_like(S2)
                        inv_S2[S2 > self.eps] = 1 / S2[S2 > self.eps]
                
                        module.__kfac_data__['U1'] = U1
                        module.__kfac_data__['U2'] = U2
                        module.__kfac_data__['V1'] = V1
                        module.__kfac_data__['V2'] = V2
                        module.__kfac_data__['inv_S1'] = inv_S1
                        module.__kfac_data__['inv_S2'] = inv_S2
                    except Exception as e:
                        module.__kfac_data__['U1'] = Variable(torch.eye(aaT.shape[0]))
                        module.__kfac_data__['V1'] = Variable(torch.eye(aaT.shape[0]))
                        module.__kfac_data__['inv_S1'] = Variable(torch.ones(aaT.shape[0]))
                        module.__kfac_data__['U2'] = Variable(torch.eye(dLdLT.shape[0]))
                        module.__kfac_data__['V2'] = Variable(torch.eye(dLdLT.shape[0]))
                        module.__kfac_data__['inv_S2'] = Variable(torch.ones(dLdLT.shape[0]))
    
                U1 = module.__kfac_data__['U1']
                U2 = module.__kfac_data__['U2']
                V1 = module.__kfac_data__['V1']
                V2 = module.__kfac_data__['V2']
                inv_S1 = module.__kfac_data__['inv_S1']
                inv_S2 = module.__kfac_data__['inv_S2']
                
                for param_name, param in module.named_parameters():
                    if type(param.grad) == type(None):
                        continue
                            
                    if param_name == 'weight':            
                        cur_grad = param.grad
                        
                        new_grad = (V1 @ torch.diag(inv_S1) @ U1.t() @ cur_grad.t() @ V2 @ torch.diag(inv_S2) @ U2.t()).t()                       
                        module.__kfac_data__['new_grad_weight'] = new_grad
                    if param_name == 'bias':
                        cur_grad = param.grad.view(param.grad.shape[0], -1)
                        module.__kfac_data__['new_grad_bias'] =  (cur_grad.t() @ V2 @ torch.diag(inv_S2) @ U2.t()).t()
                
        
        # compute nu
        nu_denom = 0
        for name, module in model.named_modules():
            module_name = module.__class__.__name__
    
            if module_name in ['Linear']:
                for param_name, param in module.named_parameters():
                    if type(param.grad) == type(None):
                        continue
                    if param_name == 'weight':
                        nu_denom += (module.__kfac_data__['new_grad_weight'] * param.grad.view(param.grad.shape[0], -1)).sum()
                    elif param_name == 'bias':
                        nu_denom += (module.__kfac_data__['new_grad_bias'] * param.grad.view(param.grad.shape[0], -1)).sum()
                    else:
                        nu_denom += (param.grad.view(param.grad.shape[0], -1) * param.grad.view(param.grad.shape[0], -1)).sum()
                        
        nu = min(1, sqrt(2 * self.delta / (abs(nu_denom.data.numpy()[0]) + self.eps)) / self.lr)
        
        # making step
        for name, module in model.named_modules():
            module_name = module.__class__.__name__
    
            if module_name in ['Linear']:
                for param_name, param in module.named_parameters():
                    if type(param.grad) == type(None):
                        continue
                    if param_name == 'weight':  
                        param.grad = nu * module.__kfac_data__['new_grad_weight']#.view(param.grad.shape)
                    elif param_name == 'bias':
                        param.grad = nu * module.__kfac_data__['new_grad_bias'].view(param.grad.shape)
            
    def step(self):
        self.compute_fisher_grads(self.model)
        self.optimizer.step()
        
        self.iter_i += 1
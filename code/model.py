import torch 
from torch import nn
from torch.nn import functional as F

### a layer to reshape the unit output for the pooling(maxout)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
 
    def forward(self, x):
        return x.view(self.shape)
    
### a single GHH layer model with nn.Module
class GHH(nn.Module):
    def __init__(self, data, d, k):
        '''
        data: the grid coordinate
          d: the dimension of the input space
          k: the number of the basis functions to be summed up
        '''
        super(GHH, self).__init__()
        self.layer = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer.weight,-1,1)
        nn.init.uniform_(self.layer.bias, -1,1)
        self.res1 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout = nn.MaxPool1d(d+1, stride = (d+1))
        self.res2 = Reshape(data.shape[0],k)
        self.eps = nn.Linear(k,1, bias = False)
        self.eps_weight = torch.randint(0,2,self.eps.weight.shape)*2.0-1.0
        self.eps.weight = nn.Parameter(self.eps_weight)
        
        
    def forward(self,x):
        output = self.layer(x)
        output = self.res1(output)
        output = self.maxout(output)
        output = self.res2(output)
        output = self.eps(output)
        return output
    
    def normalize(self, d, dim, lip_c = 1):
        output = self.layer(d)
        output = self.res1(output)
        output = self.maxout(output)
        output = self.res2(output)
        output = self.eps(output)
        output.backward(torch.ones_like(output))
        grad = d.grad
        if dim==1:
            lip = grad.abs().max()
        elif dim==2:
            lip = (grad[:,0].pow(2) + grad[:,1].pow(2)).sqrt().max()
        else:
            lip = lip_c
        self.eps.weight = nn.Parameter((lip_c/lip) * self.eps_weight)
        d.grad.data.zero_()
        
    
class GHH_01(nn.Module):
    def __init__(self, data, d, k):
        super(GHH_01, self).__init__()
        self.layer = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer.weight,-1,1)
        nn.init.uniform_(self.layer.bias, -1,1)
        self.res1 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout = nn.MaxPool1d(d+1, stride = (d+1))
        self.res2 = Reshape(data.shape[0],k)
        self.eps = nn.Linear(k,1, bias = False)
        self.eps.weight = nn.Parameter(0.1*(torch.randint(0,2,self.eps.weight.shape)*2.0-1.0))
        
    def forward(self,x):
        output = self.layer(x)
        output = self.res1(output)
        output = self.maxout(output)
        output = self.res2(output)
        output = self.eps(output)
        return output

class GHH_10(nn.Module):
    def __init__(self, data, d, k):
        super(GHH_10, self).__init__()
        self.layer = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer.weight,-1,1)
        nn.init.uniform_(self.layer.bias, -1,1)
        self.res1 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout = nn.MaxPool1d(d+1, stride = (d+1))
        self.res2 = Reshape(data.shape[0],k)
        self.eps = nn.Linear(k,1, bias = False)
        self.eps.weight = nn.Parameter(5*(torch.randint(0,2,self.eps.weight.shape)*2.0-1.0))
        
    def forward(self,x):
        output = self.layer(x)
        output = self.res1(output)
        output = self.maxout(output)
        output = self.res2(output)
        output = self.eps(output)
        return output
### a composed GHH
class composed_GHH(nn.Module):
    def __init__(self, data, d, k):
        super(composed_GHH, self).__init__()
        self.layer1 = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer1.weight,-1,1)
        nn.init.uniform_(self.layer1.bias, -1,1)
        self.layer2 = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer2.weight,-1,1)
        nn.init.uniform_(self.layer2.bias, -1,1)
        self.res11 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout1 = nn.MaxPool1d(d+1, stride = (d+1))
        self.res12 = Reshape(data.shape[0],k)
        self.eps1 = nn.Linear(k,1, bias = False)
        self.eps1.weight = nn.Parameter(torch.randint(0,2,self.eps1.weight.shape)*2.0-1.0)
        self.res21 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout2 = nn.MaxPool1d(d+1, stride = (d+1))
        self.res22 = Reshape(data.shape[0],k)
        self.eps2 = nn.Linear(k,1, bias = False)
        self.eps2.weight = nn.Parameter(torch.randint(0,2,self.eps2.weight.shape)*2.0-1.0)
        self.layer3 = nn.Linear(d, (d+1)*k)
        nn.init.uniform_(self.layer3.weight,-1,1)
        nn.init.uniform_(self.layer3.bias, -1,1)
        self.res31 = Reshape(1,data.shape[0],(d+1)*k)
        self.maxout3 = nn.MaxPool1d(d+1, stride = (d+1))
        self.res32 = Reshape(data.shape[0],k)
        self.eps3 = nn.Linear(k,1, bias = False)
        self.eps3.weight = nn.Parameter(torch.randint(0,2,self.eps3.weight.shape)*2.0-1.0)
    def forward(self, x):
        output1 = self.layer1(x)
        output1 = self.res11(output1)
        output1 = self.maxout1(output1)
        output1 = self.res12(output1)
        output1 = self.eps1(output1)
        
        output2 = self.layer2(x)
        output2 = self.res21(output2)
        output2 = self.maxout2(output2)
        output2 = self.res22(output2)
        output2 = self.eps2(output2)
        
        output = torch.cat((output1, output2), 1)
        output = self.layer3(output)
        output = self.res31(output)
        output = self.maxout3(output)
        output = self.res32(output)
        output = self.eps3(output)
        return output
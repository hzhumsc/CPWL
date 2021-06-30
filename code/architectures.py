import torch
from grid_method import *
from model import GHH

'''The 4 pre-defined architectures. '''

def arch_1(d, k, l, normalize=False):
    ghh1 = GHH(d,1,k)
    if normalize:
        ghh1.normalize(d, 1)
    res = ghh1(d)
    
    for i in range(l-1):
        ghh = GHH(res,1,k)
        if normalize:
            test = res.detach()
            test.requires_grad_(True)
            ghh.normalize(test, 1)
        res = res.clone()
        res = ghh(res)
        
    output = res.clone()
    return output


def arch_2(d, k, l, normalize = False):
    ghh1 = GHH(d,2,k)
    if normalize:
        ghh1.normalize(d, 2)
    res = ghh1(d)
    
    for i in range(l-1):
        ghh = GHH(res,1,k)
        if normalize:
            test = res.detach()
            test.requires_grad_(True)
            ghh.normalize(test, 1)
        res = res.clone()
        res = ghh(res)
        
    output = res.clone()
    return output


def arch_3(d, k, l, normalize = False):
    # start with l = 2
    ghh01 = GHH(d, 1, k)
    ghh02 = GHH(d, 1, k)
    if normalize:
        ghh01.normalize(d, 1)
        ghh02.normalize(d, 1)
    res1 = ghh01(d)
    res2 = ghh02(d)
    res = torch.cat((res1, res2), 1)
    
    for i in range(l-2):
        ghh1 = GHH(res, 2, k)
        ghh2 = GHH(res, 2, k)
        if normalize:
            test1 = res.detach()
            test2 = res.detach()
            test1.requires_grad_(True)
            test2.requires_grad_(True)
            ghh1.normalize(test1, 2)
            ghh2.normalize(test2, 2)
        res1 = res.clone()
        res2 = res.clone()
        res1 = ghh1(res1)
        res2 = ghh2(res2)
        res = torch.cat((res1, res2), 1)
        
    ghh_f = GHH(res, 2, k)
    if normalize:
        test = res.detach()
        test.requires_grad_(True)
        ghh_f.normalize(test, 2)
    res = res.clone()
    output = ghh_f(res)
    return output


def arch_4(d, k, l, normalize = False):
    ghh01 = GHH(d, 2, k)
    ghh02 = GHH(d, 2, k)
    if normalize:
        ghh01.normalize(d, 2)
        ghh02.normalize(d, 2)
    res1 = ghh01(d)
    res2 = ghh02(d)
    res = torch.cat((res1, res2), 1)
    
    for i in range(l-2):
        ghh1 = GHH(res, 2, k)
        ghh2 = GHH(res, 2, k)
        if normalize:
            test = res.detach()
            test.requires_grad_(True)
            ghh1.normalize(test, 2)
            ghh2.normalize(test, 2)
        res1 = res.clone()
        res2 = res.clone()
        res1 = ghh1(res1)
        res2 = ghh2(res2)
        res = torch.cat((res1, res2), 1)
        
    ghh_f = GHH(res, 2, k)
    if normalize:
        test = res.detach()
        test.requires_grad_(True)
        ghh_f.normalize(test, 2)
    res = res.clone()
    output = ghh_f(res)
    return output

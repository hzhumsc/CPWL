import numpy as np
import time
import random
from collections import Counter
from colorsys import hls_to_rgb
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def generate_grid(x_range, y_range, step):
    '''
    to generate the grid coordinate as a (n, d) 2d tensor,
    where n is the number of the points in the grid and d is the input dimension
    '''
    x = torch.arange(x_range[0], x_range[1]+step, step)
    y = torch.arange(y_range[1], y_range[0]-step, -1.0*step)
    
    length = x.shape[0]
    x = x.repeat(1,length).reshape((-1,1))
    y = y.repeat(length,1).transpose(0,1).reshape((-1,1))
    xy = torch.cat((x,y), 1)
    return xy, length

def compute_grad(output, data, length, pr = False):
    '''
    to compute the gradient wrt the x-axis and y-axis
    '''
    
    t1 = time.time()
    grad_x = torch.zeros(length*length)
    grad_y = torch.zeros(length*length)
    #data.grad.data.zero_()
    output.backward(torch.ones_like(output))
    grad_x = data.grad[:,0]
    grad_y = data.grad[:,1]
    grad_x = grad_x.reshape((length, length))
    grad_y = grad_y.reshape((length, length))
    t2 = time.time()
    if pr == True:
        print('elapsed time: %s ms' % ((t2 - t1)*1000))
    return grad_x, grad_y

def compute_grad_slow(output, data, length):
    '''
    to compute the gradient wrt the x-axis and y-axis with a for loop
    '''
    
    t1 = time.time()
    grad_x = torch.zeros(length*length)
    grad_y = torch.zeros(length*length)
    #data.grad.data.zero_()
    for i in range(output.shape[0]):
        loss = output[i]
        loss.backward(retain_graph=True)
    grad_x = data.grad[:,0]
    grad_y = data.grad[:,1]
    grad_x = grad_x.reshape((length, length))
    grad_y = grad_y.reshape((length, length))
    t2 = time.time()
    print('elapsed time: %s ms' % ((t2 - t1)*1000))
    return grad_x, grad_y


def get_region(grad_x, grad_y):
    '''
    distribute the regions according to the values of the gradient
    '''
    grad_x_n = grad_x.numpy()
    grad_y_n = grad_y.numpy()
    shap = grad_x_n.shape
    grad_x_n = grad_x_n.astype(str)
    _, dist_x = np.unique(grad_x_n, return_inverse = True)
    dist_x = dist_x.reshape(shap)
    grad_y_n = grad_y_n.astype(str)
    _, dist_y = np.unique(grad_y_n, return_inverse = True)
    dist_y = dist_y.reshape(shap)
    dist_x = dist_x.astype(str)
    dist_y = dist_y.astype(str)
    dist_x = np.array(dist_x, dtype = 'object')
    dist_y = np.array(dist_y, dtype = 'object')
    

    region_grad = dist_x + dist_y

    _, region_grad = np.unique(region_grad, return_inverse=True)
    region_grad = region_grad.reshape(shap)
    return region_grad, dist_x, dist_y

def get_region_fast(grad_x, grad_y):
    
    grad_x_n = grad_x.numpy()
    shape = grad_x_n.shape
    grad_x_n = grad_x_n.astype(str)
    _, dist_x = np.unique(grad_x_n, return_inverse = True)
    dist_x = dist_x.reshape(shape)
    
    return dist_x

def view_2d(x_range, y_range, step, region, save_name = None):
    '''
    plot a 2d visualization of the assigned regions
    '''
    colors = []
    n = len(Counter(region.reshape(-1)).keys())
    for i in np.arange(0, 360, 360/n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))
    random.shuffle(colors)
    
    x = np.arange(x_range[0], x_range[1]+step, step)
    y = np.arange(y_range[1], y_range[0]-step, -1.0*step)
    xx, yy = np.meshgrid(x, y, indexing = 'xy')
    grid = np.array([xx, yy])
    
    xlength = x.shape[0]
    dist = xlength/(xlength/10)
    xticks = x
    keptxticks = np.floor(xticks[::int(len(xticks)/dist)])
    xticks = ['' for x in xticks]
    xticks[::int(len(xticks)/dist)] = keptxticks
    yticks = y
    keptyticks = np.ceil(yticks[::int(len(xticks)/dist)])
    yticks = ['' for y in yticks]
    yticks[::int(len(xticks)/dist)] = keptyticks
    
#     xlength = x.shape[0]
#     dist = xlength/(xlength/10)
#     xticks = x
#     keptxticks = xticks[::int(len(xticks)/dist)]
#     xticks = ['' for x in xticks]
#     xticks[::int(len(xticks)/dist)] = keptxticks
#     yticks = y
#     keptyticks = yticks[::int(len(xticks)/dist)]
#     yticks = ['' for y in yticks]
#     yticks[::int(len(xticks)/dist)] = keptyticks
        
    plt.figure(figsize = (20,12))
    plt.axis('equal')
    ax = sns.heatmap(region, cmap = colors, yticklabels=yticks,xticklabels=xticks)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    
    
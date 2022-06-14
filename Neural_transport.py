#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author: Shiran Levy

Neural transport pyro implementation combined with deep generative models 
(SGAN or VAE) based on "" paper (references below)

========================++++++++++++++++++=====================================
MIT License

Copyright (c) 2021 Shiran Levy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
========================++++++++++++++++++=====================================

References:
    
              
"""
import argparse
import os
from os.path import exists
import json
import numpy as np
import random
import time 
import copy

# libraries for neural transport
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.distributions.transforms import iterated, affine_autoregressive, block_autoregressive
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormalizingFlow, AutoIAFNormal
from functools import partial

import dill as pickle
pickle.settings['recurse'] = False
from csv import writer

#import forward solvers
import pygimli as pg
import setup_FWsolver_torch as sfw

torch.pi = torch.acos(torch.zeros(1, dtype=torch.float64)).item() * 2

class Posterior(dist.TorchDistribution):
    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, setup):
        self.device = setup.device
        self.sigma = setup.sigma
        self.netG = setup.gen
        self.tt = setup.tt
        self.DGM = setup.DGM
        self.total_event_size = setup.total_event_size
        self.batch_size = setup.batch_size
        self.G = torch.Tensor(setup.A).to(self.device)
        self.d = torch.Tensor(setup.d).to(self.device)
        self.fwd = setup.fwd
        self.prior = prior(self.total_event_size, self.batch_size)
        self.outdir=args.outdir
        super().__init__()

    @property
    def batch_shape(self):
        '''batch_size>1 doesn't work well!'''
        return (self.batch_size,)

    @property
    def event_shape(self):
        return (self.total_event_size,)

    def sample(self, sample_shape=()):
        # it is enough to return an arbitrary sample with correct shape
        # return torch.zeros(sample_shape + self.batch_shape + self.event_shape)
        return torch.zeros(sample_shape + self.batch_shape + self.event_shape)

    def log_prob(self, state):

        '''to check: should it be multiplied by the prior?'''

        # reshape state tensor generator input shape
        if self.DGM == 'SGAN':
            zx = 5
            zy = 3
            nz = 1
            znp = torch.reshape(state, [-1, nz, zx, zy])
        else:
            znp = torch.reshape(state, [-1, self.total_event_size])
        z = znp.float().to(self.device)  # send tensor to device (cpu/gpu)

        # send the states through the generator to generate samples
        model = self.netG(z)
        if self.DGM == 'SGAN':
            model = 0.5*(model+1)  # normalize generated model to [0,1]

        if self.fwd == 'pix2pix':
            sim = model
            e = (self.d-sim).reshape([sim.shape[0], -1])
        else:
            # assign velocity to the model (0.06 and 0.08 m/ns)
            model = 0.06 + (1-model)*0.02

            s_model = 1/model[:, 0]  # convert to slowness
            if self.fwd == 'nonlinear':
#%% this block was modified from Lopez-Alvis et al. (2021) https://github.com/jlalvis/VAE_SGD/blob/master/SGD_DGM.py
                s_m = s_model.detach().cpu().numpy()
                ndata = self.tt.fop.data['t'].shape[0]
                J = np.zeros((s_m.shape[0], ndata, self.tt.inv.parameterCount))
                for ii in range(s_m.shape[0]):
                    J[ii] = nonlinear(s_m[ii], self.tt, ndata)
                self.G = torch.Tensor(J).to(self.device)
#%%  ===============================================================================================================
            sim = self.G@s_model.reshape([s_model.shape[0], -1, 1])
            e = self.d-sim[:,:,0]
            
        with open(self.outdir+'Data_RMSE.csv', 'a') as file:
            writer_object = writer(file,dialect='excel')
            b = torch.mean(e**2).clone().detach().numpy()
            writer_object.writerow([b])
            file.close()
                
        N = torch.tensor(e.shape[-1])

        log_like = - (N / torch.tensor(2.0)) * torch.log(torch.tensor(2.0) * torch.pi) - N * torch.log(self.sigma)\
            - torch.tensor(0.5) * torch.pow(self.sigma, torch.tensor(-2.0)) * torch.sum(torch.pow(e, torch.tensor(2.0)), dim=-1)
        
        #Calculating the log prior probability analytically for N(0,1):
        # log_prior =  - ( self.total_event_size / 2.0) * np.log(2.0 * torch.pi) - self.total_event_size * torch.log(torch.tensor(1)) - 0.5 * np.power(1,-2.0) * torch.sum(torch.pow(state,2.0), axis =1)

        return log_like + self.prior.log_prob(state)

def nonlinear(s_m, tt, ndata):
    G = np.zeros((ndata, tt.inv.parameterCount))
    tt.Velocity = pg.Vector(np.float64(1./s_m.flatten()))
    tt.fop.createJacobian(1./tt.Velocity)
    G = pg.utils.sparseMatrix2Dense(tt.fop.jacobian())
    return G

def prior(total_event_size, batch_size=1):
    return dist.MultivariateNormal(torch.zeros(batch_size, total_event_size, dtype=torch.float32),
                                    torch.eye(total_event_size))

def model(setup):
    pyro.sample("Z", Posterior(setup))

def guide_Normal(setup):
    loc = pyro.param('post_loc', lambda: torch.zeros(setup.total_event_size))
    scale = pyro.param('post_scale', lambda: torch.ones(setup.total_event_size))
    pyro.sample("Z", dist.MultivariateNormal(loc, scale_tril=torch.diag(scale)))  

#%%
def main(args):
    
    pyro.set_rng_seed(args.rng_seed)    #seed for reproducibility of results

    workdir = args.workdir
    outdir = args.outdir
    os.chdir(workdir)
    # data acquisition parameters
    bh_spacing = 6.5
    bh_length = 12.5
    sensor_spacing = 0.5
    ncase = 'noise'
    pix2pix = 0 if args.inversion=='data' else 1  # model inversion 1, data inversion 0
    # NeuTra parameters
    flow = args.flow_type            # IAF, NAF or BNAF, (WARNING! BNAF cannot have arbitrary values as input to posterior.log_prob())
    batch_size = 1
    num_particles = args.num_particles
    solver = args.solver    # 'linear - Straight-ray, nonlinear - Eikonal
    DGM = args.DGM  # choose from: 'VAE' or 'SGAN'
    decay_lr = 0  #If to use exponential decay of the learning rate
    setup = sfw.setup()  # initialize data object containing SGD parameters
    setup.sigma = args.sigma  # noise std [ns]

    #%% this block was modified from Lopez-Alvis et al. (2021) https://github.com/jlalvis/VAE_SGD/blob/master/SGD_DGM.py
    # check if GPU is available ()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if str(device) == 'cuda':
        cuda = True
    else:
        cuda = False

    if DGM == 'VAE':
        os.chdir('VAE')
        from autoencoder_in import VAE
        # load parameters of trained VAE
        gpath = 'VAE_inMSEeps100r1e3.pth'
        dnnmodel = VAE(cuda=cuda, gpath=gpath)
        for param in dnnmodel.parameters():
            param.requires_grad = False
        dnnmodel.to(device)
        dnnmodel.eval()
        netG = dnnmodel.decode
        os.chdir(workdir)
        samp = np.load('test_models/zinitVAE500.npy')[0]
        dims = samp.shape

    elif DGM == 'SGAN':
        os.chdir('SGAN')
        from generator import Generator
        # load parameters of trained SGAN
        gpath = 'netG_epoch_24.pth'
        dnnmodel = Generator(cuda=cuda, gpath=gpath).to(device)
        for param in dnnmodel.parameters():
            param.requires_grad = False
        dnnmodel.to(device)
        dnnmodel.eval()
        netG = dnnmodel.forward
        os.chdir(workdir)
        samp = np.load('test_models/zinitSGAN500.npy')[0]
        dims = samp.shape
    else:
        print('not a valid DGM')
    #%%  ===============================================================================================================
    
    setup.total_event_size = samp.size
    if args.saved_model=='None':
        zs = torch.reshape(pyro.sample('zs', prior(setup.total_event_size)), dims)[np.newaxis]
        if DGM == 'SGAN':
            zs = zs[np.newaxis]
    else:
        zs = torch.tensor(np.load('/home/slevy/Desktop/Neutra/neutra_SGAN/test_models/'+args.saved_model+'.npy'))

    setup.zs = zs
    m = netG(zs)
    if DGM=='SGAN':
        m = (m + 1) * 0.5
    m = m.detach().numpy()
    
    # plt.figure(1)
    # plt.imshow(m[0, 0], cmap='Greys_r')
    # plt.colorbar()
    # plt.show()
    
    #if running a model cropped from the training image uncomment the following line:
    # m = np.load('/home/slevy/Desktop/Neutra/neutra_SGAN/test_models/'+args.saved_model+'.npy')[np.newaxis,np.newaxis]
    
    setup.DGM = DGM
    setup.device = device
    setup.batch_size = batch_size
    if not pix2pix:
        if solver == 'linear':
            print('Setting up linear forward solver')
            setup = sfw.ST_tomo_fw(
                m, bh_spacing, bh_length, sensor_spacing, ncase, setup)
        else:
            print('Setting up nonlinear forward solver')
            setup = sfw.pygimli_fw(
                m, bh_spacing, bh_length, sensor_spacing, ncase, setup)

            setup.d = setup.d.array()

    else:
        setup.fwd = 'pix2pix'
        setup.d = m[0,0]
        setup.truemodel = m
    setup.gen = netG
    setup.sigma = torch.tensor(setup.sigma)
    setup.flow= flow

    # Fit an autoguide
    if flow == 'NAF':
        print("\nFitting a NAF autoguide ...")
        guide = AutoNormalizingFlow(model, partial(
            iterated, args.num_flows, affine_autoregressive))
    elif flow== 'BNAF':
        print("\nFitting a BNAF autoguide ...")
        guide = AutoNormalizingFlow(model, partial(
            iterated, args.num_flows, block_autoregressive, activation='ELU'))
    elif flow == 'IAF':
        print("\nFitting a IAF autoguide ...")
        guide = AutoIAFNormal(model, hidden_dim=[setup.total_event_size*2]*args.hidden_layers, num_transforms=args.num_flows)
    elif flow == 'normal':
        print("\nFitting Standard variational inference with a Gaussian approximation")
        guide = guide_Normal

    pyro.clear_param_store()
    
    if decay_lr==1:
        optimizer = torch.optim.Adam  # ({"lr": args.learning_rate})
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {
                                             'lr': args.learning_rate}, 'gamma': 0.1})
    else:
        scheduler = torch.optim.Adam({"lr": args.learning_rate})
        
    svi = SVI(model, guide, scheduler, Trace_ELBO(
        num_particles=num_particles, vectorize_particles=True))
    loss_vals =  []
    logS = []
    if exists(outdir+'Data_RMSE.csv'):
        os.remove(outdir+'Data_RMSE.csv') 
    start = time.time()
    for i in range(args.num_steps):
        loss = svi.step(setup)
        if (i+1)%10 == 0:
            # posterior = guide.get_posterior()
            # logS_val = -posterior.log_prob(torch.reshape(zs,(1,-1))).detach().numpy()[0]
            print("[{}]Elbo loss = {:.2f}".format(i+1, loss))
            # print("logS = {:.2f}".format(logS_val))
            # logS.append(np.int64(logS_val))
            if (i+1)<=150 and flow!='normal':
                torch.save(guide.state_dict(), outdir+'saved_NT_params_step_{:}.pt'.format(i+1))
        if (i+1)%50 == 0:
            with open(outdir+'setup.pkl', 'wb') as f:
                if setup.fwd=='nonlinear':
                    dummy = copy.copy(setup)
                    del dummy.tt
                    pickle.dump(dummy, f)
                    del dummy
                else:
                     pickle.dump(setup, f)
            with open(outdir+'loss.pkl', 'wb') as f:
                pickle.dump((loss_vals,logS), f)
        loss_vals.append(np.int64(loss))
    end = time.time()
    if flow!='normal':
        torch.save(guide.state_dict(), outdir+'saved_NT_params_step_last.pt')

    print(end - start)

    ''' save parameters'''
    pyro.get_param_store().save(outdir+'NeuTra_paramstore')
    with open(outdir+'setup.pkl', 'wb') as f:
        if setup.fwd=='nonlinear':
            del setup.tt
        pickle.dump(setup, f)
    with open(outdir+'loss.pkl', 'wb') as f:
        pickle.dump((loss_vals,logS), f)
    torch.save(guide, outdir+'Guides/saved_'+flow+'_guide_'+args.DGM+'.pt')


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.0")
    parser = argparse.ArgumentParser(description="Neutral transport and reparametrization"
    )
    parser.add_argument("--restart", default=0, type=int, help="if to restrat from existing trained model"
    )
    parser.add_argument("-n", "--num-steps", default=100, type=int, help="number of SVI steps"
    )
    parser.add_argument("-lr","--learning-rate",default=1e-2,type=float,help="learning rate for the Adam optimizer",
    )
    parser.add_argument("--rng-seed", default=23, type=int, help="RNG seed"
    )
    parser.add_argument( "--num-warmup", default=100, type=int, help="number of warmup steps for NUTS"
    )
    parser.add_argument("--num-samples",default=500,type=int,help="number of samples to be drawn from NUTS",
    )
    parser.add_argument("--num-flows", default=2, type=int, help="number of flows in the autoguide"
    )
    parser.add_argument("--solver", default= 'nonlinear', type=str, help="solver linear or nonlinear"
    )
    parser.add_argument("--flow-type", default= 'IAF', type=str, help="IAF, NAF, BNAF"
    )
    parser.add_argument("--hidden-layers", default=1, type=int, help="number of layers in each flow"
    )
    parser.add_argument("--num-particles", default=1, type=int, help="number of sample to train each step"
    )
    parser.add_argument("--DGM", default='VAE', type=str, help="Generative model VAE or SGAN"
    )
    parser.add_argument("--saved-model", default='mv1', type=str, help="upload a model from a file, 'None' if to draw randomly"
    )
    parser.add_argument("--inversion", default='data', type=str, help="type of inversion: Data or model (pix-to-pix)"
    )  
    parser.add_argument("--sigma", default=1, type=int, help="standard deviation of the noise in [ns]"
    )    
    parser.add_argument("--workdir", default='./', type=str, help="working directory (where all the models are"
    )
    parser.add_argument("--outdir", default= './', type=str, help="directory to save results in"
    )
    
    args = parser.parse_args()
    with open(args.outdir+'/run_commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    main(args)


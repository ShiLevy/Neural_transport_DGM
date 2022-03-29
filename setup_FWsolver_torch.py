#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 2021

@author Shiran Levy

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
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
import pygimli as pg
from pygimli.physics.traveltime import TravelTimeManager

from itertools import product

#%%
'''
First block modified from Lopez-Alvis et al. (2021) https://github.com/jlalvis/VAE_SGD/blob/master/DGM_inv_nonlinear_comparison.ipynb4
'''

@dataclass
class setup:
    #storing forward model parameters 
    # zinit: np.ndarray = np.zeros(1)
    sigma: float = 0.1 
    gen: nn.Module = None
    fwd: str = ''
    A: np.ndarray = np.zeros(1)
    tt: str = None
    d: np.ndarray = np.zeros(1)
    truemodel: np.ndarray = np.zeros(1)

def pygimli_fw(x_true, bh_spacing, bh_length, sensor_spacing, ncase, FWpars):

    model_true = np.copy(x_true) # shape: [1, 1, ny, nx]
    # x_true = x_true.detach().numpy()
    x = model_true.shape[-1]/10 # x in meters
    y = model_true.shape[-2]/10 # y in meters
    xcells = model_true.shape[-1]+1
    ycells = model_true.shape[-2]+1
    ############################################################
    '''Simulate synthetic traveltime data'''
    ############################################################
    
    depth = -np.arange(sensor_spacing, bh_length+0.01, sensor_spacing)
    
    sensors = np.zeros((len(depth) * 2, 2))  # two boreholes
    sensors[:len(depth), 0] = 0.0  # x
    sensors[len(depth):, 0] = bh_spacing  # x
    sensors[:, 1] = np.hstack([depth] * 2)  # y
    
    
    numbers = np.arange(len(depth))
    rays = list(product(numbers, numbers + len(numbers)))
    
    # Empty container
    scheme = pg.DataContainer()
    
    # Add sensors
    for sen in sensors:
        scheme.createSensor(sen)
    
    # Add measurements
    rays = np.array(rays)
    scheme.resize(len(rays))
    scheme.add("s", rays[:, 0])
    scheme.add("g", rays[:, 1])
    scheme.add("valid", np.ones(len(rays)))
    scheme.registerSensorIndex("s")
    scheme.registerSensorIndex("g")
    
    mygrid = pg.meshtools.createMesh2D(x = np.linspace(0.0, x, xcells), y = np.linspace(0.0, -y, ycells))
    #mygrid.createNeighbourInfos()
    print(mygrid)
    # read channels simulation:
    model_true = np.reshape(x_true,np.size(x_true))
    model_true = 0.06 + 0.02*(1-model_true) # m/ms
    mvel = model_true
    mslow = 1.0/mvel
    print(np.max(mslow))
    
    # set traveltime forward model
    ttfwd = TravelTimeManager()
    resp = ttfwd.simulate(mesh=mygrid, scheme=scheme, slowness=mslow, secNodes=2)
    ttfwd.applyData(resp)
    print(ttfwd.fop.data)
    sim_true = ttfwd.fop.data.get("t") # ns
    if ncase == 'noise':
        # add synthetic noise
        noise_lvl = FWpars.sigma
        #nnoise = noise_lvl*np.random.randn(len(sim_true))
        nnoise = np.load(os.getcwd()+'/test_models/noiserealz.npy') # ns
        sim_true = sim_true + noise_lvl*nnoise # noise_lvl scales the noise.
        # print(noise_lvl*nnoise[:10]) # ns
    
    ############################################################
    '''Set forward model'''
    ############################################################
    
    # set traveltime forward model
    tt = TravelTimeManager()
    scheme.add("t",sim_true)
    #scheme.add("err",nnoise)
    scheme.add("valid",np.ones(len(sim_true)))
    #tt.applyData(scheme)
    tt.fop.data = scheme
    tt.applyMesh(mygrid, secNodes=2)
    tt.Velocity = pg.Vector(tt.fop.mesh().cellCount(), 70.0)
    tt.inv.model = pg.Vector(tt.fop.mesh().cellCount(), 70.0)
    
    #tt.fop.setThreadCount(6) # using max number of threads causes sometimes segmentation fault
    import time
    start = time.time()
    tt.fop.createJacobian(1./tt.Velocity)
    end = time.time()
    print(end - start)
    
    fig, ax = plt.subplots()
    pg.show(mygrid, mvel, ax=ax, label="Velocity (m/ns)", showMesh=False,
            cMap='Greys_r', nLevs=3)
    tt.drawRayPaths(ax=ax, color="r", alpha=0.1)
    ax.plot(sensors[:, 0], sensors[:, 1], "ro")
    
    J = tt.fop.jacobian()
    # print("t from response:", tt.fop.response(1/tt.Velocity))
    # print("t from jacobian * model:", J.mult(1/tt.Velocity))
    
    FWpars.d = sim_true # "true" data (with or without added noise)
    FWpars.fwd = 'nonlinear' # either: 'linear' or 'nonlinear'
    FWpars.tt = tt # Nonlinear forward model provided as TravelTimeManager in PyGIMLi
    FWpars.truemodel = x_true
    # FWpars.A = J
    
    return FWpars

#%%
def ST_tomo_fw(x_true, bh_spacing, bh_length, sensor_spacing, ncase, FWpars):
    
    model_true = np.copy(x_true) # shape: [1, 1, ny, nx]
    model_true = 0.06 + 0.02*(1-model_true) # m/ms
    
    x = model_true.shape[-1]/10 # x in meters
    y = model_true.shape[-2]/10 # y in meters
    xcells = model_true.shape[-1]+1
    ycells = model_true.shape[-2]+1
    finefac = 1     # not working for channelized models
    spacing = 0.1/finefac
    nx = np.int32(model_true.shape[-1]*finefac) 
    ny = np.int32(model_true.shape[-2]*finefac)
    # x-axis is varying the fastest 
    sourcex = 0
    sourcez = np.arange(sensor_spacing, bh_length+0.01, sensor_spacing)      #sources positions in meters
    receiverx = bh_spacing      
    receiverz = np.arange(sensor_spacing, bh_length+0.01, sensor_spacing)   #receivers positions in meters
    xs = np.float32(sourcex/spacing)        #sources positions in model domain coordinates
    ys = sourcez/spacing       # divided by the spacing to get the domain coordinate                   
    rx = receiverx/spacing      #receivers positions in model domain coordinates
    rz = receiverz/spacing     # divided by the spacing to get receiverzthe domain coordinate  
    nsource = len(sourcez); nreceiver = len(receiverz)
    ndata=nsource*nreceiver
    data=np.zeros((ndata,4))
    x = np.arange(0,(nx/10)+0.1,0.1)                      
    y = np.arange(0,(ny/10)+0.1,0.1) 
    for jj in range(0,nsource):
        for ii in range(0,nreceiver):
            data[ ( jj ) * nreceiver + ii , :] = np.array([sourcex, sourcez[jj], receiverx, receiverz[ii]])
    from tomokernel_straight import tomokernel_straight_2D
    G = tomokernel_straight_2D(data,x,y)
    G = np.array(G.todense())
    s_model = 1/model_true
    sim_true = G@s_model.flatten()
    if ncase == 'noise':
        # add synthetic noise
        noise_lvl = FWpars.sigma
        #nnoise = noise_lvl*np.random.randn(len(sim_true))
        nnoise = np.load(os.getcwd()+'/test_models/noiserealz.npy') # ns
        sim_true = sim_true + noise_lvl*nnoise # noise_lvl scales the noise.
        # print(noise_lvl*nnoise[:10]) # ns
    
    # plt.imshow(model_true[0,0], cmap='gray')
    # plt.scatter(np.ones(sourcez.shape)*sourcex/spacing,sourcez/spacing, marker='*', color='r')
    # plt.scatter(np.ones(receiverz.shape)*receiverx/spacing-0.5,receiverz/spacing, marker='^', color='g')
    
    FWpars.d = sim_true # "true" data (with or without added noise)
    FWpars.fwd = 'linear' # either: 'linear' or 'nonlinear'
    FWpars.truemodel = x_true
    FWpars.A = G
    
    return FWpars
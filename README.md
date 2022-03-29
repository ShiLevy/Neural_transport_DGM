# Neural transport with deep generative models

This is a pyro implementation of the combined neural-transporta and deep generative models approach presented in "" (submitted to <> under review). 
The folder includes the SGAN and VAE .pth checkout files used to perform inversion, neural-transport inference code as well as the models presented
in the paper.

## Scripts and files:

-Neural_transport.py        --> the main routine to perform the inference and train the transform 

-setup_FWsolver_torch.py    --> setup foraward solver (either simple straight-ray or fastest path)

-autoencoder_in.py          --> VAE generation code

-generator.py               --> SGAN generation code

-test_models (folder)       --> all tested models as well as noise vector and samples to loaded for samples shape

**SGAN**

.pth files for generating SGAN images. Training codes can be found [here](https://github.com/elaloy/gan_for_gradient_based_inv/tree/master/training)

-netG_epoch_24.pth (Unmasked training for "mg" models)

-netG_epoch_34.pth (Trained with masking TI portions for the "mc" models)

**VAE**

.pth files for generating VAE images 

-VAE_inMSEeps100r1e3.pth (Taken from [here](https://github.com/jlalvis/VAE_SGD/tree/master/VAE))

## Citation :


## License:
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

## Contact:

Shiran Levy (shiran.levy@unil.ch)

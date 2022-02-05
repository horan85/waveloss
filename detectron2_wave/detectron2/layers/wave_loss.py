import torch
import torch.nn.functional as F
import numpy as np

class WaveLossWithLogits(torch.nn.Module):
    def __init__(self, valInc, spaInc, valW, spaW, normalize):
        super(WaveLossWithLogits, self).__init__()
        
        self.valInc = valInc
        self.spaInc = spaInc
        self.valW = valW
        self.spaW = spaW
        self.normalize = normalize

    def __call__(self, input, target):
        return wave_loss_with_logits(
            input=input,
            target=target,
            valInc=self.valInc,
            spaInc=self.spaInc,
            valW=self.valW,
            spaW=self.spaW,
            normalize=self.normalize
        )

# Functional definition
def wave_loss_with_logits(input, target, normalize=True, debug=False, use_bce=False):
    """
        Wave loss is an alternative to BCE for segmentation purposes.
        It is a mixture of losses that incorporates spatial and global value 
        difference between the input and the target mask.
        Args:
            input: torch logit (float32) of [-1, H, W] 
            target: torch [0, 1] in (float32) of [-1, H, W]
            valInc: float - determines the number of iterations
            spaInc: int - determines the spatial propagation speed
            valW: list or torch float tensor of length `1/valInc` of val weights
            spaW: list or torch float tensor of length `1/valInc` of spa weights
            normalize: normalize the loss with `input.numel()`
    """
    valInc = 0.01
    spaInc = 3 
    union = torch.max(input, target)
    currentWave = torch.min(input, target) # intersection
    num_iter = int(1/valInc)
    waveLoss = torch.tensor(0., device=input.device)
    
    valW = np.zeros(num_iter)
    
    valW[0] = np.log(2 * valInc)-np.log( valInc)
    for i in range(1,num_iter):
        valW[i] = np.log((i+2) * valInc)-np.log((i+1) * valInc)
    #spaW = 10 * np.flip(valW).copy()
    spaW =  np.ones_like(valW)*2
    valW = torch.from_numpy(valW)
    
    
    
    #for i in range(num_iter):
    #    spaW[i] = 0
    spaW = torch.from_numpy(spaW)

    # Initialize variables
    input = input.sigmoid()
    input = input.view(-1, input.size(-2), input.size(-1))
    target = target.view(-1, target.size(-2), target.size(-1))
    assert input.size() == target.size()

    if debug:
        valChanges = torch.empty(num_iter, dtype=torch.float32, device=input.device)
        spaChanges = torch.empty(num_iter, dtype=torch.float32, device=input.device)
        newWaves = []

    if use_bce:
        dist_fn = F.BCELoss()
    else:
        dist_fn = lambda x, y: torch.sum(torch.abs(x - y))

    # In `num_iter` iterations even points with 0. intensity reach 1. so this is
    # the maximum number that we should iterate
    for i in range(num_iter):
    
        # Loss for spatial differences
        newWave = torch.nn.functional.max_pool2d(
            input=currentWave, 
            kernel_size=[spaInc, spaInc], 
            stride=[1, 1],
            padding=[spaInc // 2, spaInc // 2]
        )

        # Compute new intersection after spatial increase
        newWave = torch.min(newWave, union)
        spatialChange = dist_fn(currentWave, newWave) * spaW[i]
        waveLoss += spatialChange
    
       
        
        currentWave2 = newWave
        
        # Loss for intensity differences
        newWave = currentWave + valInc # broadcast valInc
        newWave = torch.min(newWave, union) # elementwise min
        valueChange = dist_fn(currentWave, newWave) * valW[i]
        waveLoss += valueChange
        
        

        if debug:
            valChanges[i] = valueChange
            spaChanges[i] = spatialChange
            newWaves.append(newWave.clone())

        currentWave =  torch.max(currentWave2,newWave)

    if normalize:
        waveLoss /= input.numel()

    if debug:
        return waveLoss, valChanges, spaChanges, newWaves


    return waveLoss

def make_wave_loss_fn(cfg):
    valInc = cfg.MODEL.ROI_MASK_HEAD.WAVE_LOSS.VALINC
    spaInc = cfg.MODEL.ROI_MASK_HEAD.WAVE_LOSS.SPAINC
    valW = cfg.MODEL.ROI_MASK_HEAD.WAVE_LOSS.VALW
    spaW = cfg.MODEL.ROI_MASK_HEAD.WAVE_LOSS.SPAW
    normalize = cfg.MODEL.ROI_MASK_HEAD.WAVE_LOSS.NORMALIZE
    loss_fn = WaveLossWithLogits(valInc, spaInc, valW, spaW, normalize)
    return loss_fn

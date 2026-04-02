import sys
import argparse
import glob
import os
import math
import subprocess
import re
import time
import shutil

os.environ["LD_PRELOAD"] = "/home/gimattiolo/gits/3sixd/.venv/lib/python3.8/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0"

import torch

import numpy as np

import cv2

import CalibrationUtilities
import Utilities


def MakeRotationMatrixX(a) :
    sinA = math.sin(a)
    cosA = math.cos(a)
    R = torch.eye(3, dtype=torch.float32)
    R[1,1] = cosA
    R[1,2] = -sinA
    R[2,1] = sinA
    R[2,2] = cosA
    return R

def MakeRotationMatrixY(a) :
    sinA = math.sin(a)
    cosA = math.cos(a)
    R = torch.eye(3, dtype=torch.float32)
    R[0,0] = cosA
    R[0,2] = -sinA
    R[2,0] = sinA
    R[2,2] = cosA
    return R

def MakeRotationMatrixZ(a) :
    sinA = math.sin(a)
    cosA = math.cos(a)
    R = torch.eye(3, dtype=torch.float32)
    R[0,0] = cosA
    R[0,1] = -sinA
    R[1,0] = sinA
    R[1,1] = cosA
    return R

def MakeRotationMatrix(a,b,c) :
    sinA = math.sin(a)
    cosA = math.cos(a)
    sinB = math.sin(b)
    cosB = math.cos(b)
    sinC = math.sin(c)
    cosC = math.cos(c)

    #Rz * Ry * Rx    
    R =torch.eye(3, dtype=torch.float32)
    R[0,0] = cosB * cosC
    R[0,1] = - sinA * sinB * cosC - cosA * sinC
    R[0,2] = - cosA * sinB * cosC + sinA * sinC

    R[1,0] = cosB * sinC
    R[1,1] = - sinA * sinB * sinC + cosA * cosC
    R[1,2] = - cosA * sinB * sinC - sinA * cosC

    R[2,0] = sinB
    R[2,1] = sinA * cosB
    R[2,2] = cosA * cosB

    return R

# q is [v, s], s + vx * i + vy * j + vz * k, [u * sin(theta/2), cos(theta/2)]
def quaternion_multiplication(q1, q2) :
    # [s1v2 + s2v1 + v1 × v2, s1s2 − v1 · v2]
    v1 = q1.xyz
    v2 = q2.xyz
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q.xyz = q1.w * v2 + q2.w * v1 + np.cross(v1, v2)
    q.w = q1.w * q2.w - np.dot(v1, v2)
    return q

def quaternion_to_matrix(q) :
    s = q.w
    v = q.xyz

    # 1.0, 0.0, 0.0, # first row (not column as in GLSL!)
    # 0.0, 1.0, 0.0, # second row
    # 0.0, 0.0, 1.0  # third row
    M = np.eye(3, dtype=np.float32)

    M[0][0] = 1.0 - 2.0 * (v.y * v.y - v.z * v.z)
    M[0][1] = 2.0 * (v.x * v.y  - s * v.z)
    M[0][2] = 2.0 * (v.x * v.z - s * v.y)

    M[1][1] = 1.0 - 2.0 * (v.x * v.x - v.z * v.z)
    M[1][2] = 2.0 * (v.y * v.z - s * v.x)

    M[2][2] = 1.0 - 2.0 * (v.x * v.x - v.y * v.y)

    # because of symmetry
    M[1][0] = M[0][1]
    M[2][0] = M[0][2]
    M[2][1] = M[1][2]

    return M



class OptimizableTransform(torch.nn.Module):
    def __init__(self, angles, translation):
        super().__init__()

        self.angles = torch.nn.Parameter(data=angles, dtype=torch.float32)
        self.translation = torch.nn.Parameter(data=translation, dtype=torch.float32)

        self.M = torch.zeros(3,4, dtype=torch.float32)

    def forward(self, x):
        self.M[0:3,0:3] = MakeRotationMatrix(self.angles)
        self.M[0:3,3:] = self.translation

class OptimizableTransforms(torch.nn.Module):
    def __init__(self, angles, translations):
        super().__init__()
        n = len(angles)
        self.modules = torch.nn.ModuleList()
        for i in range(n) : 
            self.modules.append(OptimizableTransform(angles[i], translations[i]))

    def forward(self, x):
        outputs = x
        for i in range(len(self.modules)) :
            outputs = self.modules[i](outputs)
        return outputs

def BackPropagation(rank, batch_id, model, loss, scaler, optimizer) :

    # clear model param gradients after update 
    optimizer.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance
    
    if USE_SCALER :
        #with torch.autograd.detect_anomaly(True) :
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        scaler.scale(loss).backward()
    else :    
        loss.backward()

    parameters = model.parameters()
    if CHECK_GRAD :
        for p in list(parameters) :
            if p.grad is None :
                Log(rank, f'NONE GRAD!')
                break
            if utilities.IsInvalidTorchTensor(p.grad) :
                Log(rank, f'INVALID GRAD!')
                break

    if GRAD_FIXING :
        #if None, positive infinity values are replaced with the greatest finite value representable by input’s dtype
        for p in list(filter(lambda p : p.grad is not None, parameters)):
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

    if GRAD_CLIPPING :
        if USE_SCALER :
            # Unscales the gradients of optimizer's assigned params in-place
            # then clip unscaled gradient
            scaler.unscale_(optimizer)
        #torch.nn.utils.clip_grad_value_(parameters, clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=100.0, error_if_nonfinite=True)

    if USE_SCALER :
        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()
    else :    
        optimizer.step()

    global CHECK_UNUSED_PARAMS
    if batch_id == 0 and CHECK_UNUSED_PARAMS :
        can_set_static_graph = model._get_ddp_logging_data().get('can_set_static_graph')
        model_name = model.module.__class__.__name__
        GlobalLog(rank, f'{model_name}|can_set_static_graph:{can_set_static_graph}')
        GlobalLog(rank, f'{model_name}|Unused params:')
        unused = 0
        for name, param in model.named_parameters() :
            if param.grad is None:
                GlobalLog(rank, f'\t{name}')
                unused +=1
        if unused > 0 :
            Abort(rank)
            return
        GlobalLog(rank, '\tN/A')

def SetTrainMode(model, train) :
    model.train(mode=train)
    model.requires_grad_(train)

def ComputeLoss(rank, batch, outputs) :

    inputs = batch['input_data'].to(rank) 
    targets = batch['target_data'].to(rank)

    return None
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Panorama')
    
    args = parser.parse_args()

    radius = 1.0

    Intrinsics = np.zeros((3,4), dtype=np.float32)

    num_cameras = 6

    step = 2 * np.pi / num_cameras

    ExtrinsicMatrices = {}
    ExtrinsicMatrices_noisy = {}
    M_w_c = [None] * num_cameras
    M_c_w = [None] * num_cameras

    angle = 0.0
    for i in range(0, num_cameras) :

        M_w_c[i] = np.eye(4, np.float32)
        M_c_w[i] = np.eye(4, np.float32)
        
        x = math.cos(angle)
        z = math.sin(angle)
        forward = np.array([x, 0.0, z], dtype=np.float32)
        
        t_c_w = radius * forward

        down = np.array([0,1,0], dtype=np.float32)

        right = np.cros(forward, down)

        R_c_w = np.eye(3, np.float32)

        #W - > c
        M_c_w[i][0:3, 0:1] = right
        M_c_w[i][0:3, 1:2] = down
        M_c_w[i][0:3, 2:3] = forward
        M_w_c[i][0:3, 3:4] = t_c_w

        R_w_c = np.transpose(R_c_w)
        t_w_c = -np.dot(R_w_c, t_c_w) 

        M_w_c[i][0:3, 0:3] = R_w_c
        M_w_c[i][0:3, 3:4] = t_w_c


    for i in range(0, num_cameras) :
        j = (i+1)% num_cameras
        key = (i, j)
        ExtrinsicMatrices[key] = np.dot(M_w_c[j], M_c_w[i])
        ExtrinsicMatrices_noisy[key] = np.copy(ExtrinsicMatrices[key])


model = OptimizableTransforms(angles, translations)

SetTrainMode(model, train=True)

args.num_epochs = 100

rank = 'cuda0'

# Begin training 
for epoch in range(0, args.num_epochs):

    # <BEGIN EPOCH>

    log_this_epoch = epoch % args.log_period == 0
    
    epoch_start_time = time.time()

    # Set the model in training mode
    
    # avg_l1 = 0
    rank_data_loader_iterator = iter(rank_train_dataset_loader)
    rank_dataset_size = 0 

    for batch_id in range(0, len(rank_train_dataset_loader)) :

        # <BEGIN BATCH>

        # if log_this_epoch :
        #     GlobalLog(rank, f"Batch: {batch_id}/{num_rank_train_batches}")

        batch_start_time = time.time()

    
        batch = next(rank_data_loader_iterator)

        io_duration = time.time() - batch_start_time
        
        all_outputs = model(batch)
        loss =  ComputeLoss(rank, epoch, batch, all_outputs, args.model_name, discriminator, inference=False)
        BackPropagation(rank, batch_id, model, loss, scaler, optimizer)
        
        # this is the average over the batch size - the last one might have a different size than the others
        batch_train_loss = loss.item()
        rank_dataset_size += input_tensor.size()[0]
        # batch loss is the average on the batch and we need the sum to be able to compute the average on the whole dataset across all ranks
        rank_train_metrics.entries['loss'] += batch_train_loss * input_tensor.size()[0]

        batch_duration = time.time() - batch_start_time

        # <END BATCH>

    # check once if the model parameters are all used during training
    global CHECK_UNUSED_PARAMS
    if CHECK_UNUSED_PARAMS :
        CHECK_UNUSED_PARAMS = False

    if args.graphics_plot_period > 0 and epoch % args.graphics_plot_period == 0 :
        plot_id = 0
        plotter.Append(0, epoch, epoch_train_loss)
        plotter.UpdateAnimation(plot_id); plot_id +=1

        plotter.Pause()










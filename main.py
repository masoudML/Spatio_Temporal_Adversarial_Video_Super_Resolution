from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN 
from rbpn import FeatureExtractor, Discriminator
from data import get_training_set, get_eval_set, get_test_set
from tensorboardX import SummaryWriter
writer = SummaryWriter()
import torchvision
import torchvision.models as models
import math
import numpy as np

import pdb
import socket
import time
import torchvision.transforms as transforms
import cv2
import PWCNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./vimeo_septuplet/sequences')
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='RBPN_4x.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F7', help='Location to save checkpoint models')
parser.add_argument('--pretrained_disc', default='', help='sr pretrained DISC base model')
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--freeze_gen', type=bool, default=False)

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0    
    mean_discriminator_flow_adversarial_loss = 0.0
    mean_generator_flow_adversarial_loss = 0.0
    if opt.freeze_gen:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%% no GEN train %%%%%%%%%%%%%%%%%%%')
        model.eval()
    else:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%% Train %%%%%%%%%%%%%%%%%%%')
        model.train()
    iteration = 0
    avg_psnr = 0.0
    counter = 0.0
    for iteration, batch in enumerate(training_data_loader, 1):
        #input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]
        disc_flow, disc_neigbor, input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input, neigbor, flow)
        
        if opt.residual:
            prediction = prediction + bicubic


        target_left = disc_neigbor[2]
        target_right = disc_neigbor[3]

        #Discriminator Neigbor Images
        r_neigbor = []
        r_neigbor.append(target_left.cuda(1))
        r_neigbor.append(target_right.cuda(1))        
        if cuda:
            #High Resolution Real and Fake assignment for discriminator
            high_res_real = Variable(target).cuda(1)
            high_res_fake = prediction.cuda(1)
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda(1)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda(1)
        else:
            #High Resolution Real and Fake assignment for discriminator
            high_res_real = Variable(target)
            high_res_fake = prediction
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)

        #print('#################################################')
        #print(high_res_real.shape)
        #print(torch.max(high_res_real))
        #print(high_res_fake.shape)
        #print(torch.max(high_res_fake))
        #in_img = transforms.ToPILImage()(high_res_real[0].cpu())
        #in_img.save("./hd_in_img.png","PNG")
        #neighbor_img = transforms.ToPILImage()(r_neigbor[0][0].cpu())
        #neighbor_img.save("./hd_neighbor_img.png","PNG")
        #print('#################################################')
    
        #flow_left_real = get_pwc_flow(pwc_flow,high_res_real, r_neigbor[0])
        #flow_right_real = get_pwc_flow(pwc_flow,high_res_real, r_neigbor[1])
        #flow_real = torch.cat((flow_left_real,flow_right_real ),dim=1).to('cuda:1')
        
        #flow_left_fake = get_pwc_flow(pwc_flow,high_res_fake, r_neigbor[0])
        #flow_right_fake = get_pwc_flow(pwc_flow,high_res_fake, r_neigbor[1])
        
        #flow_fake = torch.cat((flow_left_fake,flow_right_fake),dim=1)
        
        #print(flow_real.shape)
        #print(flow_fake.shape)

        #print('#################################################')
            
        #print(tt.shape)
        #Discriminator Flow Images
        dflow = []
        dflow.append(disc_flow[2].cuda(1))
        dflow.append(disc_flow[3].cuda(1))

        discriminator_loss = adversarial_criterion(discriminator(high_res_real,r_neigbor,dflow), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data).cuda(1),r_neigbor,dflow), target_fake)
        #discriminator_flow_loss = adversarial_criterion(discriminator(flow_real,r_neigbor), target_real) + \
        #                     adversarial_criterion(discriminator(Variable(flow_fake.data).cuda(1),r_neigbor), target_fake)
       
        #discriminator_loss += discriminator_flow_loss

        generator_adv_loss = 5e-2*adversarial_criterion(discriminator(Variable(high_res_fake.data).cuda(1),r_neigbor,dflow), ones_const) 
        #generator_adv_flow_loss = 1e-3*adversarial_criterion(discriminator(Variable(flow_fake.data).cuda(1),r_neigbor), ones_const)
        
        #generator_adv_loss += generator_adv_flow_loss
        
        target_norm = (target/torch.max(target)).to('cuda:1')
        prediction_norm = (prediction/torch.max(prediction)).to('cuda:1')

        real_features = Variable(feature_extractor(target_norm).data).cuda(1)
        fake_features = feature_extractor(prediction_norm)
        
        vgg_loss = 0.006*content_criterion(fake_features, real_features)

        loss = 0.001*criterion(high_res_fake, high_res_real) + vgg_loss + generator_adv_loss
        t1 = time.time()

        #--- VGG/Context Loss
        mean_generator_content_loss += vgg_loss.data

        #--- Gen Av Loss
        mean_generator_adversarial_loss += generator_adv_loss.data #  generator_adv_loss

        #--- Disc Adv Loss
        mean_discriminator_loss += discriminator_loss.data

        # flow adv losses 
        #mean_discriminator_flow_adversarial_loss += discriminator_flow_loss.data
        #mean_generator_flow_adversarial_loss += generator_adv_flow_loss.data

        epoch_loss += loss.data
        if not opt.freeze_gen:
            print('$$$$$$$$$$$$$$$$$$$$$ backward $$$$$$$$$$$$$$$$$$$')
            loss.backward()
            optimizer.step()
        else:
            print('$$$$$$$$$$$$$$$$$$$$$ No backward $$$$$$$$$$$$$$$$$$$')


        discriminator_loss.backward()
        nn.utils.clip_grad_norm_(discriminator.parameters(), 5)
        optim_discriminator.step()

        avg_batch_psnr = PSNR(target[0], prediction[0])
        if opt.batchSize > 1:
            avg_psnr2 = PSNR(target[1], prediction[1])
            avg_psnr += (avg_batch_psnr+avg_psnr2)/2.0
    
        avg_psnr += avg_batch_psnr
        counter +=1

        #if iteration % 10 ==0:
        #    avg_psnr = eval(model)
        #    if avg_psnr > best_psnr:
        #        print('Save Best Model PSNR: '+ str(avg_psnr))
        #        best_psnr = avg_psnr
        #        checkpoint(epoch)
        #    model.train()

        
        #print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data[0], (t1 - t0)))
        print('#########################################################################')
        print("===> Epoch[{}]({}/{}): PSNR: {:.4f},Loss: {:.4f}, VGG Loss {:.8f}, genAdv Loss {:.8f}, discAdv Loss {:.8f}  || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader),avg_batch_psnr,loss.data,vgg_loss, generator_adv_loss,discriminator_loss,  (t1 - t0)))
#    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f} Avg. Disc Loss {:.8f}".format(epoch, epoch_loss / len(training_data_loader),mean_discriminator_loss/len(training_data_loader)))
    writer.add_scalar('data/avg_epoch_loss', epoch_loss/len(training_data_loader), epoch)
    writer.add_scalar('data/avg_gen_vgg_loss', mean_generator_content_loss/len(training_data_loader), epoch)
    writer.add_scalar('data/avg_gen_adv_loss', mean_generator_adversarial_loss/len(training_data_loader), epoch)
    writer.add_scalar('data/avg_disc_adv_loss', mean_discriminator_loss/len(training_data_loader), epoch)
    #writer.add_scalar('data/avg_disc_flow_adv_loss', mean_discriminator_flow_adversarial_loss/len(training_data_loader), epoch)
    #writer.add_scalar('data/avg_gen_flow_adv_loss', mean_generator_flow_adversarial_loss/len(training_data_loader), epoch)

    return (avg_psnr/counter)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_gen_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    disc_model_out_path = opt.save_folder+str(opt.upscale_factor)+'x_disc_'+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    torch.save(discriminator.state_dict(), disc_model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border, :]
    imdff = pred - gt
    rmse = torch.sqrt(torch.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def get_pwc_flow(model, im1,im2):
    #im1 = 1.0 * im1/255.0
    #im2 = 1.0 * im2/255.0

    flow_input = torch.cat((im1,im2), dim=1)
    flow_input = flow_input.to(torch.device('cuda:0'))
    model.eval()
    flow_neighbor_pwc = model(flow_input)
    flow_neighbor_pwc = 20 * nn.Upsample(scale_factor=4, mode='bilinear')(flow_neighbor_pwc)
    
    objectOutput = open('./pwc_flow.flo', 'wb')
    tens = flow_neighbor_pwc[0]
    #tens = flow[i][0]
    np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objectOutput)
    np.array([tens.size(2), tens.size(1)], np.int32).tofile(objectOutput)
    np.array(tens.cpu().detach().numpy().transpose(1, 2, 0), np.float32).tofile(objectOutput)
    objectOutput.close()

    return flow_neighbor_pwc


def eval(model,predicted, target):
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor, flow)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
       #  save_img(prediction.cpu().data, str(count), True)

        #save_img(target, str(count), False)

        #prediction=prediction.cpu()
        #prediction = prediction.data[0].numpy().astype(np.float32)
        #prediction = prediction*255.

        #target = target.squeeze().numpy().astype(np.float32)
        #target = target*255.

        #psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        #avg_psnr_predicted += psnr_predicted
        count+=1
        target = target.cuda(0)
        psnr = PSNR(prediction, target)
        avg_psnr_predicted += psnr

    return (avg_psnr_predicted/count)

def eval_test(model):
    model.eval()
    count=1
    avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        input, target, neigbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(input, neigbor, flow)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
       #  save_img(prediction.cpu().data, str(count), True)

        #save_img(target, str(count), False)

        #prediction=prediction.cpu()
        #prediction = prediction.data[0].numpy().astype(np.float32)
        #prediction = prediction*255.

        #target = target.squeeze().numpy().astype(np.float32)
        #target = target*255.

        #psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        #avg_psnr_predicted += psnr_predicted
        count+=1
        target = target.cuda(0)
        psnr = PSNR(prediction, target)
        avg_psnr_predicted += psnr

    return (avg_psnr_predicted/count)



cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
#test_set = get_eval_set(opt.test_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
#testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)

print('===> Loading datasets')
test_set = get_test_set(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.file_list, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)


print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor) 
    if opt.freeze_gen:
        print('Freezing Generator')
        for param in model.parameters():
            param.requires_grad = False
        
    discriminator = Discriminator().cuda(1) #1


feature_extractor = FeatureExtractor(models.vgg19(pretrained=True).cuda(1))
#print(feature_extractor)
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()
ones_const = Variable(torch.ones(opt.batchSize, 1).cuda(1))

#pwc_flow = PWCNet.__dict__['pwc_dc_net']("./PWCNet/pwc_net.pth.tar").cuda(0)

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')
print(opt.pretrained)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    dmodel_name = os.path.join(opt.pretrained_disc)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')
    if os.path.exists(dmodel_name):
        print('RESTORE: DISCRIMINATOR Path does exist!')
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        discriminator.load_state_dict(torch.load(dmodel_name, map_location=lambda storage, loc: storage),strict=False)
        print('Pre-trained Disc model is loaded.')


if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.lr)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    avg_psnr = train(epoch)
    print('#########################################################################')
    print('Epoch avg PSNR :' + str(avg_psnr))
    #test()

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)

# export scalar data to JSON for external processing
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
from torchvision.transforms import *
import torch.nn.functional as F
from dbpns import Net as DBPNS



class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #self.feat0_flow = ConvBlock(4, 128, 3, 2, 1, activation='prelu', norm='batch')
        self.feat0 = ConvBlock(3, 128, 3, 2, 1, activation='prelu', norm='batch')
        self.feat01 = ConvBlock(128, 256, 3, 2, 1, activation='prelu', norm='batch')

        self.feat1 = ConvBlock(8, 5, 3, 1, 1, activation='prelu', norm='batch')
        #self.feat1_flow = ConvBlock(5, 5, 3, 1, 1, activation='prelu', norm='batch')
        self.feat2 = ConvBlock(5, 3, 3, 1, 1, activation='prelu', norm='batch')

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256) #2,256,64,64
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.convmerge = nn.Conv2d(512,256 ,3, stride=1, padding=1)
        self.bnmerge = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256,256 ,3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(256)


        # Replaced original paper FC layers with FCN
        self.conv_nin = nn.Conv2d(256, 1, 1, stride=1, padding=1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def swish(self,x):
        return x * F.sigmoid(x)

    def forward(self, target,neigbor,flow=None):
        #All 3 inputs - target, neigbor and flow are 4x scale images
        input_channels = target.shape[1]//2
        if flow is None:
            feat_input = self.feat0_flow(target)
        else:
            feat_input = self.feat0(target)
        feat_input = self.feat01(feat_input)

        for j in range(len(neigbor)):

            #print('********* SHAPES *******')
            #print('shape of target is',target.shape)
            #print('shape of feat_input is',feat_input.shape)
            #print('shape of neigbor is',neigbor[j].shape)
            #print('shape of flow is',flow[j].shape)
            #print('********* END SHAPES *******')
            #feat_frame.append(self.feat1(torch.cat((x.float(), neigbor[j].float(), flow[j].float()),1)))
            if flow is None:
                targ = target[:,(input_channels*j):(input_channels*(j+1)),:,:]
                x = self.feat1_flow(torch.cat((targ.float(), neigbor[j].float()),1))
            else:
                x = self.feat1(torch.cat((target.float(), neigbor[j].float(), flow[j].float()),1))
            #print('shape after feat1',x.shape)
            x = self.feat2(x)
            #print('shape after feat2',x.shape)
            #Pretraining for these layers is available
            x = self.swish(self.conv1(x))
            #print('shape after conv1',x.shape)
            x = self.swish(self.bn2(self.conv2(x)))
            #print('shape after conv2',x.shape)
            x = self.swish(self.bn3(self.conv3(x)))
            #print('shape after conv3',x.shape)
            x = self.swish(self.bn4(self.conv4(x)))
            #print('shape after conv4',x.shape)
            x = self.swish(self.bn5(self.conv5(x)))
            #print('shape after conv5',x.shape)
            x = self.swish(self.bn6(self.conv6(x)))
            #print('shape after conv6',x.shape)
            x = self.swish(self.bn7(self.conv7(x)))
            #print('shape after conv7',x.shape)
            x = self.swish(self.bn8(self.conv8(x)))
            #print('shape of output of recurrence 1/conv8 is',x.shape)
            #print('shape of feat_input is',feat_input.shape)
            feat_input = self.swish(self.bnmerge(self.convmerge(torch.cat((feat_input,x),1))))
            feat_input = self.swish(self.bn9(self.conv9(feat_input)))
        out = self.conv_nin(feat_input)
        return F.sigmoid(F.avg_pool2d(out, out.size()[2:])).view(out.size()[0], -1)


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(Net, self).__init__()
        #base_filter=256
        #feat=64
        self.nFrames = nFrames
        
        if scale_factor == 2:
        	kernel = 6
        	stride = 2
        	padding = 2
        elif scale_factor == 4:
        	kernel = 8
        	stride = 4
        	padding = 2
        elif scale_factor == 8:
        	kernel = 12
        	stride = 8
        	padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)
                
        #Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)
        
        #Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)
        
        #Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)
        
        #Reconstruction
        self.output = ConvBlock((nFrames-1)*feat, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    torch.nn.init.kaiming_normal_(m.weight)
        	    if m.bias is not None:
        		    m.bias.data.zero_()
            
    def forward(self, x, neigbor, flow):
        ### initial feature extraction
        feat_input = self.feat0(x)
        feat_frame=[]
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]),1)))
        
        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)
            h1 = self.res_feat1(feat_frame[j])
            
            e = h0-h1
            e = self.res_feat2(e)
            h = h0+e
            Ht.append(h)
            feat_input = self.res_feat3(h)
        
        ####Reconstruction
        out = torch.cat(Ht,1)        
        output = self.output(out)
        
        return output

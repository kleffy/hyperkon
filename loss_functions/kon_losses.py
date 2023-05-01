import numpy as np
# from numpy import dot
# from numpy.linalg import norm

import torch
from torch.nn import functional as F

class SimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(SimilarityLoss, self).__init__()

    def forward(self, anchor, positives):

        # compute similarity
        sim = anchor @ positives.transpose(-2, -1)

        # divide by a small temperature and take softmax
        sim2 = sim / 1e-6
        prob = sim2.softmax(dim=-1)

        # initialise labels - this gives the position of the correct labels
        labels = torch.arange(anchor.shape[0])

        # squeeze dimensions
        prob = prob.squeeze()
        labels = labels.squeeze()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # take cross entropy loss
        loss = F.cross_entropy(prob, labels.to(device))

        # print(f'===== running loss ===== \n{loss}')
        return loss


class SimilarityLoss2(torch.nn.Module):
    def __init__(self, ):
        super(SimilarityLoss2, self).__init__()

    def forward(self, anchor, positives):

        # compute similarity
        sim = anchor @ positives.transpose(-2, -1)

        # divide by a small temperature and take softmax
        sim2 = sim / 0.1
        prob = sim2.softmax(dim=-1)

        # initialise labels - this gives the position of the correct labels
        labels = torch.arange(anchor.shape[0])

        # squeeze dimensions
        prob = prob.squeeze()
        labels = labels.squeeze()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # take cross entropy loss
        loss = F.cross_entropy(prob, labels)

        # print(f'===== running loss ===== \n{loss}')
        return loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, ):
        super(ContrastiveLoss, self).__init__()


    def forward(self, queries, keys, temperature = 0.1):
        b, device = queries.shape[0], queries.device
        logits = queries @ keys.t()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits /= temperature
        return F.cross_entropy(logits, torch.arange(b, device=device))


class NTXentLoss(torch.nn.Module):
    def __init__(self, ):
        super(NTXentLoss, self).__init__()

        
    def forward(self, queries, keys, temperature = 0.1):
        b, device = queries.shape[0], queries.device

        n = b * 2
        projs = torch.cat((queries, keys))
        logits = projs @ projs.t()

        mask = torch.eye(n, device=device).bool()
        logits = logits[~mask].reshape(n, n - 1)
        logits /= temperature

        labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        loss /= n
        return loss

# if __name__ == "__main__":
    # cos_sim = (a @ b.T) / (norm(a)*norm(b))
    # from model.vgg_pl import Vgg16Net_PL
    # from dataloaders.hyper_dataset import HyperImageDataSet
    # from utils.img_util import HyperImageUtility

    # data_dir = r'/vol/research/RobotFarming/Projects/data/full'
    # enmap = HyperImageDataSet(root_dir=data_dir, is_train=True, apply_augmentation=True)
    # _, a, a_hat = enmap.__getitem__(2)

    # img_util = HyperImageUtility()
    # img_data = img_util.move_axis(a[3])
    # img_data = img_util.extract_percentile_range(img_data, 2, 98)
    # rgb_img = img_util.extract_rgb(data=img_data, r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))

    # img_data2 = img_util.move_axis(a_hat[5])
    # img_data2 = img_util.extract_percentile_range(img_data2, 2, 98)
    # rgb_img2 = img_util.extract_rgb(data=img_data2, r_range=(46, 48), g_range=(23, 25), b_range=(8, 10))

    # # imgs = [torch.from_numpy(rgb_img), torch.from_numpy(rgb_img_a_hat)]
    # # imgs = torchvision.utils.make_grid(imgs)
    # img_util.display_image_1x1(rgb_img, rgb_img2)


    # b = 32; c=88#; h=32; w=32
    # num_classes=128
    # learning_rate=0.001
    # # size = (b, c)
    # # anchor = torch.rand(size=size)
    # # positives = torch.rand(size=size)
    # net = Vgg16Net_PL(in_channels=c, out_features=num_classes, learning_rate=learning_rate)
    # anchor = net(torch.from_numpy(a).float())
    # positives = net(torch.from_numpy(a_hat).float())
    # anchor = F.normalize(anchor)
    # positives = F.normalize(positives)
    # # print(anchor.shape)
    # sl = SimilarityLoss()
    # loss, sim, prob = sl.forward(anchor, positives)


    # print('========== Vector ==========')
    # print(anchor)
    # print('========== Similarity ==========')
    # print(sim)
    # print('========== Probabilities ==========')
    # print(prob)
    # print('========== LOSS ==========')
    # print(loss)
    
        

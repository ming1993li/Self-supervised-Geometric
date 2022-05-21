from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, EquivarianceConstraintLoss, DeepSupervision, OFPenalty
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import Logger, RankLogger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.rotation_utils import randomly_rotate_images, randomly_rotate_images_3
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
from synbn.sync_batchnorm import convert_model
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    from apex.multi_tensor_apply import multi_tensor_applier
    use_apex = True
except ImportError:
    # raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    use_apex = False

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    global use_apex
    if not args.use_apex:
        use_apex = False
    # torch.autograd.set_detect_anomaly(True)
    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    if args.evaluate and args.load_weights:
        save_dir = os.path.dirname(args.load_weights)
    else:
        save_dir = './logs/' + str(datetime.datetime.now())[:19]
    if args.evaluate:
        if args.target_names[0] == 'vehicleID':
            log_name = 'log_test_{}.txt'.format(args.test_size)
        else:
            log_name = 'log_test.txt'
    else:
        log_name = 'log_train.txt'
    sys.stdout = Logger(osp.join(save_dir, log_name))
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, test_size=args.test_size, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu, args=args)
    if args.syn_bn:
        print('Using synchronized batch normalization!')
        model = convert_model(model)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.evaluate and args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_htri = TripletLoss(margin=args.margin)
    if args.use_rotation_prediction:
        criterion_rot = CrossEntropyLoss(num_classes=args.rotation, use_gpu=use_gpu, label_smooth=args.lable_smooth_for_rot)
    else:
        criterion_rot = None
    if args.use_equivariance_constraint:
        assert args.rotation == 4, 'Equivariance Constraint should only be used when rotation is equal to 4!'
        criterion_eqv = EquivarianceConstraintLoss(mode='kl_divergence', use_gpu=use_gpu)
    else:
        criterion_eqv = None
    if args.use_of_penalty:
        criterion_of = OFPenalty(args.of_beta)
    else:
        criterion_of = None

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if use_apex:
        model.cuda()
        # Apex FP16 training
        print("Using Mix Precision training")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    else:
        model = nn.DataParallel(model).cuda() if use_gpu else model

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=None, use_apex=use_apex)
        for _ in range(args.start_epoch):
            scheduler.step()

    if args.evaluate:
        print('Evaluate only')

        if args.flipped_test:
            flipped_test = True
        else:
            flipped_test = False

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))

            if name == 'veri' or name == 'market1501' or name == 'dukemtmc' or name == 'aicity20':
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1, mAP_i2i, mAP_i2t, distmat, cmc = test(model, queryloader, galleryloader, use_gpu, epoch=-1,
                                                             save_dir=save_dir, flipped_test=flipped_test)
                print(mAP_i2t, mAP_i2i, cmc[0], cmc[4])
            elif name == 'vehicleID':
                cmc1 = []
                cmc5 = []
                repeat = 1
                # repeat = 10
                for i in range(repeat):
                    queryloader = testloader_dict[name]['query']
                    galleryloader = testloader_dict[name]['gallery']
                    rank1, mAP_i2i, mAP_i2t, distmat, cmc = test(model, queryloader, galleryloader, use_gpu,
                                                                 flipped_test=flipped_test)
                    cmc1.append(cmc[0])
                    cmc5.append(cmc[4])
                    if i < repeat - 1:
                        dm = ImageDataManager(use_gpu, test_size=args.test_size, **dataset_kwargs(args))
                        trainloader, testloader_dict = dm.return_dataloaders()

                print(np.mean(cmc1), np.mean(cmc5), cmc1)
            else:
                raise ValueError('The target dataset {} is not supported!'.format(args.target_names[0]))

        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')

    test_now = False
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, criterion_xent, criterion_htri, criterion_rot, criterion_eqv, criterion_of, optimizer, trainloader, use_gpu)
        scheduler.step()
        if (epoch + 1) < args.max_epoch - 2 * args.step_epoch:
            if (epoch + 1) % 5 == 0:
                test_now = True
        else:
            # if (epoch + 1) % 2 == 0 or (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.max_epoch:
                test_now = True
        if test_now:
            test_now = False
            print('=> Test')
            for name in args.target_names:
                print('Evaluating {} ...'.format(name))
                queryloader = testloader_dict[name]['query']
                galleryloader = testloader_dict[name]['gallery']
                rank1, mAP_i2i, mAP_i2t, _, _ = test(model, queryloader, galleryloader, use_gpu, epoch + 1, save_dir)
                ranklogger.write(name, epoch + 1, rank1, mAP_i2i, mAP_i2t)

            if epoch + 1 > args.max_epoch - 3 * args.step_epoch:
                if use_apex:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'rank1': rank1,
                        'mAP_i2i': mAP_i2i,
                        'mAP_i2t': mAP_i2t,
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict(),
                    }, save_dir)
                else:
                    save_checkpoint({
                        'state_dict': model.state_dict(),
                        'rank1': rank1,
                        'mAP_i2i': mAP_i2i,
                        'mAP_i2t': mAP_i2t,
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'optimizer': optimizer.state_dict(),
                    }, save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


def train(epoch, model, criterion_xent, criterion_htri, criterion_rot, criterion_eqv, criterion_of, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    xent_losses_rs = AverageMeter()
    htri_losses = AverageMeter()
    htri_losses_rs = AverageMeter()
    rot_losses = AverageMeter()
    rot_accs = AverageMeter()

    eqv_losses = AverageMeter()
    of_losses = AverageMeter()
    accs = AverageMeter()
    accs_rs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    # for p in model.parameters():
    #     p.requires_grad = True    # open all layers

    end = time.time()
    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):
        data_time.update(time.time() - end)

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        if args.use_rotation_prediction and epoch > args.rot_start_epoch:
            rot_imgs, rot_label = randomly_rotate_images(copy.deepcopy(imgs))
            cls_score_rot, kp_map_rot = model(rot_imgs, pids, rot=True)
            outputs, features, kp_map, of_features = model(imgs, pids, rot=False)
        else:
            cls_score_rot, kp_map_rot = None, None
            outputs, features, kp_map, of_features = model(imgs, pids, rot=False)
        # rotation image space
        if args.use_rot_space:
            outputs_rs, features_rs, _, _ = model(randomly_rotate_images_3(copy.deepcopy(imgs)), pids, rot=False)

        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids, args.lambda_kp_global)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids, args.lambda_kp_global)
        else:
            htri_loss = criterion_htri(features, pids)

        if args.use_rot_space:
            if isinstance(outputs_rs, (tuple, list)):
                xent_loss_rs = DeepSupervision(criterion_xent, outputs_rs, pids, args.lambda_kp_global)
            else:
                xent_loss_rs = criterion_xent(outputs_rs, pids)

            if isinstance(features_rs, (tuple, list)):
                htri_loss_rs = DeepSupervision(criterion_htri, features_rs, pids, args.lambda_kp_global)
            else:
                htri_loss_rs = criterion_htri(features_rs, pids)

        if args.use_rotation_prediction and epoch > args.rot_start_epoch:
            # if isinstance(cls_score_rot, (tuple, list)):
            #     rot_loss = DeepSupervision(criterion_rot, cls_score_rot, rot_label)
            # else:
            rot_loss = criterion_rot(cls_score_rot, rot_label)

        if args.use_equivariance_constraint:
            eqv_loss = criterion_eqv(kp_map, kp_map_rot, rot_label)

        if args.use_of_penalty and epoch > args.start_of_penalty:
            of_loss = criterion_of(of_features)

        if args.use_rotation_prediction and epoch > args.rot_start_epoch:
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss + args.lambda_rot * rot_loss
        else:
            loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss

        if args.use_equivariance_constraint:
            loss += args.lambda_eqv * eqv_loss

        if args.use_of_penalty and epoch > args.start_of_penalty:
            loss += args.lambda_of * of_loss

        if args.use_rot_space:
            loss += args.lambda_xent_rs * xent_loss_rs + args.lambda_htri_rs * htri_loss_rs

        optimizer.zero_grad()

        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))
        accs.update(accuracy(outputs, pids)[0])
        if args.use_rotation_prediction and epoch > args.rot_start_epoch:
            rot_losses.update(rot_loss.item(), rot_label.size(0))
            rot_accs.update(accuracy(cls_score_rot, rot_label)[0])

        if args.use_equivariance_constraint:
            eqv_losses.update(eqv_loss.item(), pids.size(0))

        if args.use_of_penalty and epoch > args.start_of_penalty:
            of_losses.update(of_loss.item(), pids.size(0))
        if args.use_rot_space:
            xent_losses_rs.update(xent_loss_rs.item(), pids.size(0))
            htri_losses_rs.update(htri_loss_rs.item(), pids.size(0))
            accs_rs.update(accuracy(outputs_rs, pids)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Xent_rs {xent_rs.val:.4f} ({xent_rs.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Htri_rs {htri_rs.val:.4f} ({htri_rs.avg:.4f})\t'
                  'Rotl {rotl.val:.4f} ({rotl.avg:.4f})\t'
                  'Eqvl {eqvl.val:.10f} ({eqvl.avg:.10f})\t'
                  'Ofpl {ofpl.val:.10f} ({ofpl.avg:.10f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                  'Acc_rs {acc_rs.val:.2f} ({acc_rs.avg:.2f})\t'
                  'Racc {racc.val:.2f} ({racc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                xent_rs=xent_losses_rs,
                htri=htri_losses,
                htri_rs=htri_losses_rs,
                rotl=rot_losses,
                eqvl=eqv_losses,
                ofpl=of_losses,
                acc=accs,
                acc_rs=accs_rs,
                racc=rot_accs
            ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, epoch=0, save_dir=None, ranks=[1, 5, 10, 20], flipped_test=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        if args.use_rotation_prediction:
            cls_score_rots, rot_labels = [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            if args.use_rotation_prediction:
                rot_imgs, rot_label = randomly_rotate_images(copy.deepcopy(imgs))
                cls_score_rot, _ = model(rot_imgs, pids, rot=True)
                end = time.time()
                features, _ = model(imgs, pids, rot=False)
                batch_time.update(time.time() - end)
                if flipped_test:
                    flipped_features, _ = model(torch.flip(imgs, dims=(3,)), pids, rot=False)
                    features = (features + flipped_features) / 2.0
            else:
                cls_score_rot = None
                features, _ = model(imgs, pids, rot=False)
                if flipped_test:
                    flipped_features, _ = model(torch.flip(imgs, dims=(3,)), pids, rot=False)
                    features = (features + flipped_features) / 2.0

            if args.use_rotation_prediction:
                cls_score_rots.append(cls_score_rot.data.cpu())
                rot_labels.append(rot_label.data.cpu())

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        if args.feat_norm:
            qf = F.normalize(qf, dim=1, p=2)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        if args.use_rotation_prediction:
            cls_score_rots = torch.cat(cls_score_rots, dim=0)
            rot_labels = torch.cat(rot_labels, dim=0)
            rot_acc = accuracy(cls_score_rots, rot_labels)[0]
            print('Rotation degree prediction accuracy in query set: \t {}'.format(rot_acc))

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        if args.use_rotation_prediction:
            cls_score_rots, rot_labels = [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            if args.use_rotation_prediction:
                rot_imgs, rot_label = randomly_rotate_images(copy.deepcopy(imgs))
                cls_score_rot, _ = model(rot_imgs, pids, rot=True)
                end = time.time()
                features, _ = model(imgs, pids, rot=False)
                batch_time.update(time.time() - end)
                if flipped_test:
                    flipped_features, _ = model(torch.flip(imgs, dims=(3,)), pids, rot=False)
                    features = (features + flipped_features) / 2.0
            else:
                cls_score_rot = None
                features, _ = model(imgs, pids, rot=False)
                if flipped_test:
                    flipped_features, _ = model(torch.flip(imgs, dims=(3,)), pids, rot=False)
                    features = (features + flipped_features) / 2.0

            if args.use_rotation_prediction:
                cls_score_rots.append(cls_score_rot.data.cpu())
                rot_labels.append(rot_label.data.cpu())

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        if args.feat_norm:
            gf = F.normalize(gf, dim=1, p=2)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        if args.use_rotation_prediction:
            cls_score_rots = torch.cat(cls_score_rots, dim=0)
            rot_labels = torch.cat(rot_labels, dim=0)
            rot_acc = accuracy(cls_score_rots, rot_labels)[0]
            print('Rotation degree prediction accuracy in gallery set: \t {}'.format(rot_acc))

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    if len(args.target_names) == 1 and args.target_names[0]== 'veri' and args.image2track_test:
        gallery_lines = open(args.gallery_name_list).readlines()
        track_lines = open(args.gallery_track_list).readlines()

        gallery_names = []
        for gallery_line in gallery_lines:
            gallery_names.append(gallery_line.strip())

        track_pids = []
        track_camids = []
        track_feats = []
        for i, track_line in enumerate(track_lines):
            line = track_line.strip()
            track_pids.append(int(line.split(' ')[0].split('_')[0]))
            track_camids.append(int(line.split(' ')[0].split('_')[1].split('c')[1]))
            images = line.split(' ')[1:]
            track_indexes = []
            for _, image in enumerate(images):
                idx = gallery_names.index(image.strip())
                track_indexes.append(idx)
            track_feat = gf[track_indexes, :]
            track_feat = torch.mean(track_feat, dim=0, keepdims=True)
            track_feats.append(track_feat)
        trackf = torch.cat(track_feats, dim=0)

        track_pids = np.asarray(track_pids)
        track_camids = np.asarray(track_camids)

        m, n = qf.size(0), trackf.size(0)
        distmat_i2t = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(trackf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat_i2t.addmm_(1, -2, qf, trackf.t())
        distmat_i2t = distmat_i2t.numpy()

        print('Image2track test CMC and mAP')
        _, mAP_i2t = evaluate(args.target_names[0], distmat_i2t, q_pids, track_pids, q_camids, track_camids, args.max_rank)

        m, n = qf.size(0), gf.size(0)
        distmat_i2i = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat_i2i.addmm_(1, -2, qf, gf.t())
        distmat_i2i = distmat_i2i.numpy()

        print('Image2image test CMC and mAP')
        cmc, mAP_i2i = evaluate(args.target_names[0], distmat_i2i, q_pids, g_pids, q_camids, g_camids, args.max_rank)

        print('Results ----------')
        print('Image2track mAP: {:.4%}'.format(mAP_i2t))
        print('Image2image mAP: {:.4%}'.format(mAP_i2i))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.4%}'.format(r, cmc[r - 1]))
        print('------------------')

        return cmc[0], mAP_i2i, mAP_i2t, distmat_i2i, cmc
    else:
        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        cmc, mAP = evaluate(args.target_names[0], distmat, q_pids, g_pids, q_camids, g_camids, args.max_rank)

        print('Results ----------')
        print('mAP: {:.4%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.4%}'.format(r, cmc[r - 1]))
        print('------------------')

        # if return_distmat:
        #     return distmat
        return cmc[0], mAP, 0.0, distmat, cmc


if __name__ == '__main__':
    main()

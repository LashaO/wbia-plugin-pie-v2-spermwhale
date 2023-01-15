# -*- coding: utf-8 -*-
import os.path as osp
import time
import numpy as np
import torch
from .engine import Engine
from metrics import compute_distance_matrix
from utils.reidtools import visualize_ranked_results
from utils.avgmeter import AverageMeter
from torch.nn import functional as F

from metrics import eval_onevsall


class PIEEngine(Engine):
    """Engine class for learning PIE for animal re-identification."""

    def __init__(self, datamanager, use_gpu=True):
        super(PIEEngine, self).__init__(datamanager, use_gpu)

    def test(
        self,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir="",
        ranks=[1, 5, 10, 20],
        rerank=False,
        visrank_resize=True,
    ):
        r"""Tests model on target datasets."""
        self.set_model_mode("eval")

        name = self.datamanager.source
        rank1, mAP = self._evaluate(
            dataset_name=name,
            test_loader=self.test_loader,
            dist_metric=dist_metric,
            normalize_feature=normalize_feature,
            visrank=visrank,
            visrank_topk=visrank_topk,
            save_dir=save_dir,
            ranks=ranks,
            rerank=rerank,
            visrank_resize=visrank_resize,
        )

        if self.writer is not None:
            self.writer.add_scalar(f"Test/{name}/rank1", rank1, self.epoch)
            self.writer.add_scalar(f"Test/{name}/mAP", mAP, self.epoch)

        return rank1

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name="",
        test_loader=None,
        dist_metric="euclidean",
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir="",
        ranks=[1, 5, 10, 20],
        rerank=False,
        visrank_resize=True,
    ):
        batch_time = AverageMeter()

        def _feature_extraction(data_loader):
            f_, pids_ = [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()
                end = time.time()
                features = self.extract_features(imgs)
                batch_time.update(time.time() - end)
                features = features.data.cpu()
                f_.append(features)
                pids_.extend(pids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            return f_, pids_

        print("Extracting features from query set ...")
        qf, q_pids = _feature_extraction(test_loader)
        print("Done, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        print("Speed: {:.4f} sec/batch".format(batch_time.avg))

        if normalize_feature:
            print("Normalzing features with L2 norm ...")
            qf = F.normalize(qf, p=2, dim=1)

        print("Computing distance matrix with metric={} ...".format(dist_metric))
        distmat = compute_distance_matrix(qf, qf, dist_metric)
        distmat = distmat.numpy()

        print("Computing CMC and mAP ...")
        cmc, mAP = eval_onevsall(distmat, q_pids)

        #
        print("** Results **")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            self.wandb.log({"Rank-{:<3}".format(r): cmc[r - 1]})

        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.test_set,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, "visrank_" + dataset_name),
                topk=visrank_topk,
                resize=visrank_resize,
            )

        return cmc[0], mAP

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1
import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.sqrt(np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)) / ground_truth.shape[0] / ground_truth.shape[1]
    performance['mae'] = np.sum(np.abs((prediction - ground_truth) * mask)\
        / np.sum(mask)) / ground_truth.shape[0] / ground_truth.shape[1]
    performance['mape'] = np.sum(np.abs((prediction - ground_truth) / (ground_truth + 1e-6) * mask)\
        / np.sum(mask)) / ground_truth.shape[0] / ground_truth.shape[1]
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0
    bt_longall = 1.0
    bt_rand = 1.0
    bt_rand5 = 1.0
    bt_rand10 = 1.0
    day_results = np.zeros((prediction.shape[1], 8))
    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            #if mask[cur_rank][i] < 0.5:
            #    continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
            
        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        pre_topall = set()
        pre_rand1 = set()
        pre_rand5 = set()
        pre_rand10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            #if mask[cur_rank][i] < 0.5:
            #    continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
            pre_topall.add(cur_rank)
        
        sid = np.random.randint(prediction.shape[0], size=1)
        pre_rand1.add(rank_pre[sid[0]])
        sid = np.random.choice(prediction.shape[0], 5, replace=False)
        for id in sid:
            pre_rand5.add(rank_pre[id])
        sid = np.random.choice(prediction.shape[0], 10, replace=False)
        for id in sid:
            pre_rand10.add(rank_pre[id])

        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if False: #mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt
        day_results[i,0] = mrr_top

        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top
        day_results[i,1] = bt_long

        real_ret_rat_rand = ground_truth[list(pre_rand1)[0]][i]
        bt_rand += real_ret_rat_rand
        day_results[i,5] = bt_rand

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5
        day_results[i,2] = bt_long5

        real_ret_rat_rand5 = 0
        for pre in pre_rand5:
            real_ret_rat_rand5 += ground_truth[pre][i]
        real_ret_rat_rand5 /= 5
        bt_rand5 += real_ret_rat_rand5
        day_results[i,6] = bt_rand5

        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10
        day_results[i,3] = bt_long10

        real_ret_rat_rand10 = 0
        for pre in pre_rand10:
            real_ret_rat_rand10 += ground_truth[pre][i]
        real_ret_rat_rand10 /= 10
        bt_rand10 += real_ret_rat_rand10
        day_results[i,7] = bt_rand10

        # back testing on all
        real_ret_rat_topall = 0
        for pre in pre_topall:
            real_ret_rat_topall += ground_truth[pre][i]
        real_ret_rat_topall /= prediction.shape[0]
        bt_longall += real_ret_rat_topall
        day_results[i,4] = bt_longall


    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    performance['btl5'] = bt_long5
    performance['btl10'] = bt_long10
    performance['btlall'] = bt_longall
    performance['btr'] = bt_rand
    performance['btr5'] = bt_rand5
    performance['btr10'] = bt_rand10
    np.savetxt('day_results.csv', day_results, delimiter=',')
    return performance

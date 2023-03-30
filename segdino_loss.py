import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import torch.distributed as dist
import torch
import math

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


def init_msn_loss(num_views=1, tau=0.1, me_max=True, return_preds=False):
    """
    Make unsupervised patch-based MSN loss

    :num_views: number of anchor views
    :param tau: cosine similarity temperature
    :param me_max: whether to perform me-max regularization
    :param return_preds: whether to return anchor predictions
    """
    # softmax = torch.nn.Softmax(dim=1)
    softmax = torch.nn.Softmax(dim=2)

    def sharpen(p, T):
        sharp_p = p**(1./T)
        sharp_p_sum = torch.sum(sharp_p, dim=1, keepdim=True)
        sharp_p /= sharp_p_sum
        return sharp_p

    def snn(query, supports, support_labels, temp=tau):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query) # 默认的计算是2范数
        supports = torch.nn.functional.normalize(supports) # 默认的计算是2范数
        tem = query @ supports.T / temp
        softmax_tem = softmax(tem)
        re = softmax_tem @ support_labels
        # sumtem = softmax_tem.sum(dim=1)
        sumtem = softmax_tem.sum(dim=2)
        return re

    def loss(anchor_views, target_views, prototypes, proto_labels, T=0.25,
             use_entropy=False, use_sinkhorn=False, sharpen=sharpen, snn=snn):

        # Step 1: compute anchor predictions
        probs = snn(anchor_views, prototypes, proto_labels)
        probssumdim1 = probs.sum(dim=1)
        probssumdim2 = probs.sum(dim=2)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            # targets = sharpen(snn(target_views, prototypes, proto_labels), T=T)
            targets = snn(target_views, prototypes, proto_labels)
            # print(probs.size(), targets.size(),num_views)

            if use_sinkhorn:
                targets = distributed_sinkhorn(targets)
            # targets = torch.cat([targets for _ in range(num_views)], dim=0)

        # print(probs.size(),targets.size())

        # Step 3: compute cross-entropy loss H(targets, queries)
        probs_log = torch.log(probs**(-targets))
        # probs_sum = torch.sum(probs_log, dim=1)
        probs_sum = torch.sum(probs_log, dim=2)
        loss = torch.mean(probs_sum)

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            probs_mean = torch.mean(probs, dim=0)
            probs_mean2 = torch.mean(probs_mean, dim=0)
            avg_probs = AllReduce.apply(probs_mean2)

            avg_probs_log = torch.log(avg_probs**(-avg_probs))
            avg_probs_len = len(avg_probs)

            avg_probs_len_log= math.log(float(avg_probs_len))
            avg_probs_log_sum = - torch.sum(avg_probs_log)

            rloss = avg_probs_log_sum + avg_probs_len_log

        sloss = 0.
        if use_entropy:
            pp_log = torch.log(probs**(-probs))
            pp_log_sum = torch.sum(pp_log, dim = 2)
            sloss = torch.mean(pp_log_sum)

        # print(targets.size())
        # -- loggings
        with torch.no_grad():
            targets_argmax = targets.argmax(dim=1)
            # num_ps = float(len(set(targets.argmax(dim=1).tolist())))
            num_ps = None
            max_t = targets.max(dim=1).values.mean()
            min_t = targets.min(dim=1).values.mean()
            log_dct = {'np': num_ps, 'max_t': max_t, 'min_t': min_t}

            targets_max_mean = targets.max(dim=1).values
            targets_min_mean = targets.min(dim=1).values

        if return_preds:
            return loss, rloss, sloss, log_dct, targets

        return loss, rloss, sloss, log_dct

    return loss


@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    # continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf


class BPDAattack(object):
    def __init__(self, model=None, defense=None, device=None, epsilon=None, test = None, learning_rate=0.5,
                 max_iterations=100, clip_min=0, clip_max=1):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.defense = defense
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.device = device
        self.test = test

    def perturb(self, x, y):
        """
        Given examples (X_nat, y), returns their adversarial
        counterparts with an attack length of epsilon.
        """

        adv = x.detach().clone().cuda(0)

        lower = np.clip(x.detach().cpu().numpy() - self.epsilon, self.clip_min, self.clip_max)
        upper = np.clip(x.detach().cpu().numpy() + self.epsilon, self.clip_min, self.clip_max)

        for i in range(self.MAX_ITERATIONS):
            with tf.device('/gpu:2'):
                adv_purified = self.defense(adv,self.test)
            adv_purified = adv_purified.cuda(0)
            y = y.cuda(0)
            adv_purified.requires_grad_()
            adv_purified.retain_grad()

            scores = self.model(adv_purified)
            loss = self.loss_fn(scores, y)
            loss.backward()

            grad_sign = adv_purified.grad.data.sign()
            adv += self.LEARNING_RATE * grad_sign

            adv_img = np.clip(adv.detach().cpu().numpy(), lower, upper)
            # adv = torch.Tensor(adv_img).to(self.device)
            adv = torch.Tensor(adv_img).cuda(0)
        return adv
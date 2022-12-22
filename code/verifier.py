import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from deeppoly import DeepPolyVerifier

DEVICE = 'cpu'
DTYPE = torch.float32


'''
    net:
     
    inputs:
        - b stands for batch := 1
        - MNIST: [b, 1, 28, 28]
        - CIFAR-10: [b, 3, 32, 32]
    
    true_label:
        - the ground truth label of inputs
        - integer in the range [0, 9] denoting the handwritten digit
        
    lower bound l:
        - l = inputs - eps
    
    upper bound u:
        - u = inputs + eps
    
    output: return 1 iff we can show that in [l, u] the network only outputs true_label else return 0
            ==> The network outputs true_label if the output vector at index true_label has the biggest value
            
            We need to check that the lower bound of the output vector at position true_label is the maximum value in
            the output vector ("verified"). If this is not the case, we showed that the network is "not verified".
'''


def analyze(net, inputs, eps, true_label, N_iters=500, lrate=1, early_stop=10):
    DEBUG = False

    if DEBUG:
        print(net)
    # Make sure we are not changing the network parameters
    net.requires_grad_(False)

    # Boolean array which labels are already verified
    verified_vector = torch.zeros(10, dtype=torch.bool)
    verified_vector[true_label] = True

    softmax = torch.nn.Softmax(dim=0)

    # The initialization of the alpha values
    # alpha_values = [0, 0.25, 0.5,  1, -1]
    alpha_values = [-1]
    optimizer = None
    grad_vars = []

    verifiers = []
    for alpha_init in alpha_values:
        verifiers.append(DeepPolyVerifier(net.modules(), true_label=true_label, loss_layer=True, alpha_strategy=alpha_init, DEBUG=DEBUG))

    # Thetas for linear combination of the different approximations (Not used anymore)
    thetas_lb = torch.rand(len(verifiers), 10) * 2 - 1.0  # Bring thetas in range [-1, 1]
    thetas_lb.requires_grad = True
    thetas_ub = torch.rand(len(verifiers), 10) * 2 - 1.0  # Bring thetas in range [-1, 1]
    thetas_ub.requires_grad = True
    # grad_vars.append(thetas_lb)
    # grad_vars.append(thetas_ub)

    last_loss = torch.tensor(float("inf"))
    randomize_alphas_in = early_stop

    # for verify_label in to_verify:
    for itr in range(N_iters):
        total_lb = None
        total_ub = None

        # Set sum of thetas for neuron i to 1 with the softmax function
        t_lb = torch.nn.functional.softmax(thetas_lb, dim=0)
        t_ub = torch.nn.functional.softmax(thetas_ub, dim=0)

        for i, verifier in enumerate(verifiers):
            lb, ub = verifier(inputs, eps)
            if optimizer is None:
                for vars in verifier.get_grad_vars():
                    grad_vars.append(vars)

            if DEBUG:
                l_out = softmax(lb)
                u_out = softmax(ub)
                out = softmax(net(inputs))

                if itr == 0:
                    print("ub:", torch.round(u_out[true_label] * 100).item(), "lb:",
                          torch.round(l_out[true_label] * 100).item(), "network out:",
                          torch.round(out[0][true_label] * 100).item())

                    lower_out, upper_out, net_out = torch.argmax(l_out, dim=0).item(), torch.argmax(u_out,
                                                                                                    dim=0).item(), torch.argmax(
                        out, dim=1).item()
                # print("uc:", torch.round(nn.functional.softmax(uc, dim=0) * 100)[true_label].item(), "lc:", torch.round(nn.functional.softmax(lc, dim=0) * 100)[true_label].item())

            verify_test = lb[true_label] - ub
            verify_test[true_label] = 1
            if verify_test.min() > 0:
                return 1  # We verified the image

            # Linearly combining the outputs if we have several networks. This is problematic for the resnet ones
            # since training several Verifiers at the same time is not in the 1 minute time limit
            # lb = t_lb[i] * lb
            # ub = t_ub[i] * ub

            # In this manner total_lb and total_ub could just be lb and ub
            if total_lb is None:
                total_lb = lb
                total_ub = ub
            else:
                total_lb += lb
                total_ub += ub

        if optimizer is None:
            optimizer = torch.optim.AdamW(params=grad_vars, lr=lrate)
            # optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
            # optimizer = torch.optim.SGD(params=grad_vars, lr=lrate)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        # We are interested in the lower bound of the following loss
        # Loss := sum(ReLU(- (x_true - x_i))) for x_i != x_true
        # Since we are taking - x_i we need to take the upper bounds for the x_i´s
        # loss = total_lb[true_label] - total_ub  # [10]
        loss = total_lb
        verified_vector = torch.logical_or(verified_vector, loss > 0).detach()

        # If all our constraints are fulfilled, we have verified the network
        if not False in verified_vector:
            return 1

        # We do not care about the value of the true label
        loss[verified_vector] = 1.
        # We are interested in the entries where the lower constraint of true_label is smaller than the upper constraint
        # of an other label i.e. loss[j] < 0 ==> - loss[j] > 0 and the rest can be set to zero (using ReLU)

        loss = torch.nn.functional.relu(-loss)
        loss = loss.sum()  # .log()
        # loss = (loss * loss).sum()  # .log()

        if DEBUG:
            print(loss.item())
        optimizer.zero_grad()
        loss.backward()

        # Adding some noise to the gradients in the hope to overcome local minimas
        for var in grad_vars:
            rand_noise = torch.rand(var.shape)
            var.grad += rand_noise * 5e-7
        optimizer.step()
        # scheduler.step()

        # If we see that we are stuck in a minima try to redistribute th alpha values and restart training
        if loss >= last_loss:
            randomize_alphas_in -= 1
            if randomize_alphas_in < 0:
                for index in range(len(grad_vars)):
                    grad_vars[index].data = (torch.rand(grad_vars[index].size()) * 5 - 5)
                randomize_alphas_in = early_stop
                last_loss = torch.tensor(float("inf"))
        else:
            last_loss = loss
            randomize_alphas_in = early_stop

    return 0


def main():
    # Just skeleton code
    net, inputs, eps, true_label = None
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()

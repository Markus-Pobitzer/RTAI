import torch
# from torch import *
from networks import *
import numpy as np


class DeepPolyVerifier(nn.Module):
    """
    DeepPoly Verifier, given a network an input and an epsilon, it approximates lower and upper bounds of the output
    neurons.
    """

    def __init__(self, modules, true_label, loss_layer=False, alpha_strategy=0, DEBUG=False):
        """
        Initializes a DeepPoly Verifier that tries to verify the given network stored in modules.
        :param modules: the network to analyze as a list of torch modules
        :param alpha_strategy: the initialization strategy of the introduced alphas (learnable parameters). If
                               alpha_strategy is between 0 and 1 set it to the given value, else initialize randomly.
        :param DEBUG: flag for debugging
        """
        super(DeepPolyVerifier, self).__init__()

        # The modules containing all the layers of the network to analyze
        self.modules = list(modules)
        self.DEBUG = DEBUG
        # Reference to all the alphas, i.e. the parameters to train
        self.grad_vars = []
        # Just the zero constant as a tensor
        self.zero = torch.tensor([0.])
        # The true label
        self.true_label = true_label

        # If we want to add a loss layer:
        self.loss_layer = loss_layer

        self.first_layer = None
        self.last_layer = None
        self.layer_list = []
        self.alpha_strategy = alpha_strategy

        # Constructing the verifier
        self.setup()

    def setup(self):
        prev = None  # Verifier(name="Input Layer") # Only used for the test

        # Used to skip some layers that are already in the ResNetVerifier contained
        skip = 0
        next_skip = 0
        # self.first_layer = prev
        for module in self.modules:

            current = None
            skip += next_skip
            next_skip = 0
            if isinstance(module, Normalization):
                current = NormalizationFlattenVerifier(module)
            elif isinstance(module, nn.Linear):
                current = LinearVerifier(module)
            elif isinstance(module, nn.ReLU):
                current = ReLUVerifier(module, alpha_strategy=self.alpha_strategy)
            elif isinstance(module, nn.Conv2d):
                current = ConvolutionVerifier(module)
            elif isinstance(module, BasicBlock):
                current = ResNetVerifier(module, true_label=self.true_label, DEBUG=self.DEBUG)
                # Dunno about a better wy than getting rid of the layers that are already in the Basic block
                next_skip += len(current.children)
                if self.DEBUG:
                    print("Skipping the next", next_skip, "layers")
            elif isinstance(module, nn.Identity):
                current = IdentityVerifier(module)
            elif isinstance(module, nn.BatchNorm2d):
                current = BatchNormVerifier(module)

            if current is not None:
                if skip > 0:
                    skip -= 1
                    next_skip = 0
                    if self.DEBUG:
                        print("Removed", module)
                else:
                    if prev is None:
                        self.first_layer = current
                    else:
                        prev.next = current
                        current.prev = prev
                    prev = current
                    self.layer_list.append(current)

        if self.loss_layer:
            # Add the los layer
            current = FinalLossVerifier(self.true_label)
            prev.next = current
            current.prev = prev
            self.layer_list.append(current)

        self.last_layer = self.layer_list[-1]

        if self.DEBUG:
            print(" =================== Verifier ================== ")
            current = self.first_layer
            while current:
                print(current.name)
                current = current.next
            print(" ===================   END    ================== ")

    def get_grad_vars(self):
        self.grad_vars = []
        for layer in self.layer_list:
            if isinstance(layer, ReLUVerifier):
                self.grad_vars.append(layer.alphas)
            elif isinstance(layer, ResNetVerifier):
                for var in layer.get_grad_vars():
                    self.grad_vars.append(var)
        return self.grad_vars

    def forward(self, inputs, eps, clip_min=0., clip_max=1., backpropagation=True):
        # We have to calculate the bounds every time we call forward
        lower_input = torch.clamp(inputs - eps, min=clip_min, max=clip_max)
        upper_input = torch.clamp(inputs + eps, min=clip_min, max=clip_max)

        return self.first_layer.forward(lower_input, upper_input)


class Verifier(nn.Module):
    def __init__(self, prev=None, next=None, name="Verifier Module"):
        super(Verifier, self).__init__()
        # The previous layer in the network, relevant for the backsubstitution
        self.prev = prev
        # The next layer in the network, relevant for the forward (i.e. box) calculation
        self.next = next

        # The lower and upper bounds of this layer
        self.lb = None
        self.ub = None

        # DPConstraint containing lc, uc, lc_bias, uc_bias
        self.constraint = None

        self.zero = torch.tensor([0.])

        # self.return_constraint = False
        self.name = name

    def forward(self, lb, ub):
        # Given bounds [l, u] and weight w to multiply
        # where w < 0:
        #   l_new = u * w
        #   u_new = l * w
        # else:
        #   l_new = l * w
        #   u_new = u * w
        #
        # To calculate the given bounds we need to distinguish between positive and negative weights
        self.lb = lb @ self.constraint.positive_lc().T + ub @ self.constraint.negative_lc().T + self.constraint.lc_bias
        self.ub = ub @ self.constraint.positive_uc().T + lb @ self.constraint.negative_uc().T + self.constraint.uc_bias

        # To make sure lb <= ub
        assert not (True in torch.gt(self.lb, self.ub)), "Forward: lb > ub in " + self.name

        back_lb, back_ub, _ = self.backsubstitution()

        # Debug:
        debug(self.lb, self.ub, back_lb, back_ub, self.name)

        # See if we get a tighter bound
        # self.lb = torch.max(back_lb, self.lb)
        # self.ub = torch.min(back_ub, self.ub)
        # Maybe we just need to take the value from the backsubstitution ... ?
        self.lb = back_lb
        self.ub = back_ub

        # To make sure lb <= ub
        if True in torch.gt(self.lb, self.ub):
            # Otherwise the debugger reaches the recursion depth
            self.next = None
            self.prev = None
            a = 0
        assert not (True in torch.gt(self.lb, self.ub)), "lb > ub in " + self.name

        if self.next is not None:
            # Detach lb and ub from computational graph just to be sure
            # self.lb = self.lb.detach()
            # self.ub = self.ub.detach()
            return self.next.forward(self.lb, self.ub)
        else:
            return self.lb, self.ub

    def backsubstitution(self, prev_constraints=None, constraints_only=False):
        constraint = None
        if self.constraint is None:
            # We have nothing to do in this case
            if self.prev is None and prev_constraints is None:
                return self.lb, self.ub, constraint
            # We need to look at previous layers but we do not need to compute new constraints
            constraint = prev_constraints
        elif prev_constraints is None:
            # We are in the last layer of the current network
            constraint = self.constraint.transposed_constraint()
            # Check if there is only one layer int the network, and return if so
            if self.prev is None:
                return self.lb, self.ub, constraint
        else:
            # Compute the bias of the upper constraint up until now
            uc_bias = self.constraint.uc_bias @ prev_constraints.positive_uc() + \
                      self.constraint.lc_bias @ prev_constraints.negative_uc() + prev_constraints.uc_bias

            # Compute the upper constraint up until now
            uc = self.constraint.uc.T @ prev_constraints.positive_uc() + \
                 self.constraint.lc.T @ prev_constraints.negative_uc()

            # Compute the bias of the lower constraint up until now
            lc_bias = self.constraint.lc_bias @ prev_constraints.positive_lc() + \
                      self.constraint.uc_bias @ prev_constraints.negative_lc() + prev_constraints.lc_bias

            # Compute the lower constraint up until now
            lc = self.constraint.lc.T @ prev_constraints.positive_lc() + \
                 self.constraint.uc.T @ prev_constraints.negative_lc()

            constraint = DPConstraints(lc, uc, lc_bias, uc_bias)

        if self.prev is None:
            if constraints_only:
                return None, None, constraint
            else:
                # print(self.name, self.constraint is None, prev_constraints is None, constraints_only)
                # We are the first layer and compute the optimized bounds
                # Computing the lower (lb) and upper (ub) bounds based on the current backsubstitution step
                lb = self.lb @ constraint.positive_lc() + self.ub @ constraint.negative_lc() + constraint.lc_bias
                ub = self.ub @ constraint.positive_uc() + self.lb @ constraint.negative_uc() + constraint.uc_bias
                return lb, ub, constraint
        else:
            # There are layers before the current one
            return self.prev.backsubstitution(constraint, constraints_only=constraints_only)


def debug(lb, ub, back_lb, back_ub, name=""):
    """
    Just a debug fct that we do not track the doubly linked structure
    :param lb:
    :param ub:
    :param back_lb:
    :param back_ub:
    :param name:
    :return:
    """
    a = 0


class NormalizationFlattenVerifier(Verifier):
    def __init__(self, module, prev=None, next=None):
        super(NormalizationFlattenVerifier, self).__init__(prev, next, "NormalizationFlattenVerifier")
        self.module = module

    def forward(self, lb, ub):
        # We assume that Normalization is the first function. Afterwards we directly flatten the input
        self.lb = torch.flatten(self.module(lb))
        self.ub = torch.flatten(self.module(ub))

        if self.next is not None:
            # self.backsubstitution()
            return self.next.forward(self.lb, self.ub)
        # else:
        # return self.backsubstitution()


class IdentityVerifier(Verifier):
    def __init__(self, module, prev=None, next=None):
        super(IdentityVerifier, self).__init__(prev, next, "IdentityVerifier")
        self.module = module

    def forward(self, lb, ub):
        weights = torch.diag(torch.ones(lb.size()))  # weights of the Linear layer
        bias = torch.zeros(lb.size())  # bias of the Linear layer

        self.lb = lb @ weights.T
        self.ub = ub @ weights.T

        # To make sure lb <= ub
        assert not (True in torch.gt(self.lb, self.ub)), "Forward: lb > ub in " + self.name

        # self.next = None
        # self.prev = None
        self.constraint = DPConstraints(weights, weights, bias, bias)

        back_lb, back_ub, _ = self.backsubstitution()
        # See if we get a tighter bound
        self.lb = torch.max(back_lb, self.lb)
        self.ub = torch.min(back_ub, self.ub)

        # To make sure lb <= ub
        assert not (True in torch.gt(self.lb, self.ub)), "lb > ub in " + self.name

        if self.next is not None:
            # Maybe detach lb and ub from computational graph just to be sure
            # self.lb = self.lb.detach()
            # self.ub = self.ub.detach()
            return self.next.forward(self.lb, self.ub)
        else:
            return self.lb, self.ub


class LinearVerifier(Verifier):
    def __init__(self, module, prev=None, next=None):
        super(LinearVerifier, self).__init__(prev, next, "LinearVerifier")
        self.module = module
        self.weights = self.module.weight.detach()  # weights of the Linear layer
        self.bias = self.module.bias.detach()  # bias of the Linear layer
        self.constraint = DPConstraints(self.weights, self.weights, self.bias, self.bias)


class ReLUVerifier(Verifier):
    def __init__(self, module, prev=None, next=None, alpha_strategy=0.):
        super(ReLUVerifier, self).__init__(prev, next, "ReLUVerifier")
        self.module = module
        self.alphas = None
        self.alpha_strategy = alpha_strategy

    def forward(self, lb, ub):
        # Analyze the ReLU function

        # Case where ub <= 0 (strictly negative)
        # ub_neg = torch.where(ub <= 0, ub, zero)
        # lb_neg = torch.where(ub <= 0, lb, zero)

        # Case where lb >= 0 (strictly positive)
        # ub_pos = torch.where(lb >= 0, ub, zero)
        # lb_pos = torch.where(lb >= 0, lb, zero)
        value_pos = torch.where(lb >= 0, torch.tensor([1.]), self.zero)

        # Case where lb < 0 and ub > 0 (crossing ReLU)
        ub_crossing = torch.where(lb < 0, ub, self.zero).where(ub > 0, self.zero)
        lb_crossing = torch.where(lb < 0, lb, self.zero).where(ub > 0, self.zero)

        # Computing slope and make sure not to divide by zero
        slope = torch.where(ub_crossing > 0, ub_crossing / (ub_crossing - lb_crossing), self.zero)

        # New upper constraints
        upper_diag = torch.diag(slope + value_pos)

        # Get the trainable parameter i.e. alphas
        if self.alphas is None:
            if 0 <= self.alpha_strategy <= 1:
                self.alphas = (torch.ones(slope.size()) * self.alpha_strategy).detach()
                # self.alphas = (torch.ones(slope.size()) * - torch.inf).detach()
            else:
                # Initialize alphas to [-1. 0]
                self.alphas = (torch.rand(slope.size()) - 1.).detach()
            self.alphas.requires_grad = True

        # Create local alphas variable to keep self.alphas as leaf nodes (needed for optimization)
        # alphas = self.alphas.clamp(0, 1).where(slope > 0, self.zero)
        alphas = torch.sigmoid(self.alphas).where(slope > 0, self.zero)
        # New lower constraints
        lower_diag = torch.diag(alphas + value_pos)

        # Compute the upper and lower bound, keep in mind that we approximate the lower bound!
        self.lb = lb @ lower_diag
        self.ub = torch.max(self.zero, ub)

        zero_vector = torch.zeros(slope.size())
        # zero_matrix = torch.diag(zero_vector)
        upper_bias = slope * (-1) * lb_crossing
        # upper_constraint <= slope * (x - lb) <= slope * (previous_upper_constraint - lb)
        # lower_constraint >= alpha * x >= alpha * previous_lower_constraint
        self.constraint = DPConstraints(lower_diag, upper_diag, zero_vector, upper_bias)

        back_lb, back_ub, _ = self.backsubstitution()

        # Debug:
        debug(self.lb, self.ub, back_lb, back_ub, self.name)

        # See if we get a tighter bound
        # self.lb = torch.max(back_lb, self.lb)
        # self.ub = torch.min(back_ub, self.ub)
        # Maybe we just need to take the value from the backsubstitution ... ?
        self.lb = back_lb
        self.ub = back_ub

        # To make sure lb <= ub
        assert not (True in torch.gt(self.lb, self.ub)), "lb > ub in " + self.name

        if self.next is not None:
            # Maybe detach lb and ub from computational graph just to be sure
            # self.lb = self.lb.detach()
            # self.ub = self.ub.detach()
            return self.next.forward(self.lb, self.ub)
        else:
            return self.lb, self.ub


class ConvolutionVerifier(Verifier):
    def __init__(self, module, prev=None, next=None):
        super(ConvolutionVerifier, self).__init__(prev, next, "ConvolutionVerifier")
        self.module = module

    def setup(self, lb, ub):

        # parameters
        weights = self.module.weight.detach()
        kernel_size = self.module.kernel_size[0]
        padding = self.module.padding[0]
        stride = self.module.stride[0]
        in_channels = self.module.in_channels
        out_channels = self.module.out_channels

        # compute all dimensions
        in_features = lb.view(-1).size()[0]
        in_height = int(np.sqrt(in_features / in_channels))
        in_width = int(np.sqrt(in_features / in_channels))
        in_width_p = in_width + padding * 2
        in_height_p = in_height + padding * 2
        out_height = int((in_height + padding * 2 - kernel_size) / stride + 1)
        out_width = int((in_width + padding * 2 - kernel_size) / stride + 1)
        out_features = out_channels * out_height * out_width

        size_p = in_height_p * in_width_p
        in_dim = size_p * in_channels
        out_dim = out_height * out_width * out_channels
        res = torch.zeros((out_dim, in_dim))

        # build row fillers
        len_rows = (in_channels - 1) * size_p + (kernel_size - 1) * in_width_p + kernel_size
        rows = torch.zeros((out_channels, len_rows))
        channels = torch.zeros((out_channels, len_rows))

        for i_out in range(out_channels):
            for i_in in range(in_channels):
                i_p = i_in * size_p
                for k in range(kernel_size):
                    start = i_p + k * in_width_p
                    end = start + kernel_size
                    rows[i_out, start:end] = weights[i_out, i_in, k]
                    channels[i_out, start:end] = weights[i_out, i_in, k]

            for i_out_height in range(out_height):
                for i_out_width in range(out_width):
                    start = i_out_height * stride * in_width_p + i_out_width * stride
                    end = start + len_rows
                    output = i_out * out_height * out_width + i_out_height * out_width + i_out_width
                    res[output, start:end] = channels[i_out]

        # remove padding
        padding_rows = []
        for i_in in range(in_channels):
            for i_in_height in range(in_height_p):
                for i_in_width in range(in_width_p):
                    if i_in_width < padding or i_in_width >= padding + in_width:
                        padding_rows.append(i_in * size_p + i_in_height * in_width_p + i_in_width)

                if i_in_height < padding or i_in_height >= padding + in_height:
                    start = i_in * size_p + i_in_height * in_width_p
                    end = start + in_width_p
                    padding_rows = padding_rows + list(range(start, end))

        padding_rows = list(np.unique(np.array(padding_rows)))  # delete duplicates

        lc = torch.from_numpy(np.delete(res.numpy(), padding_rows, axis=1)).detach()

        if self.module.bias is None:
            ret_bias = torch.zeros(out_features)
        else:
            ret_bias = torch.repeat_interleave(self.module.bias.detach(), out_width * out_height)

        self.constraint = DPConstraints(lc, lc, ret_bias, ret_bias)  # store constraints

    def forward(self, lb, ub):
        # First call to forward, we need to set up weight matrix
        if self.constraint is None:
            self.setup(lb, ub)

        return super().forward(lb, ub)


class ResNetVerifier(Verifier):
    def __init__(self, module, true_label, prev=None, next=None, DEBUG=False):
        super(ResNetVerifier, self).__init__(prev, next, "ResNetVerifier")
        self.module = module
        self.alphas = []
        self.children = []
        # Keep track of the path a and b
        self.path_a = None
        self.path_b = None

        self.first_forward = True
        self.DEBUG = DEBUG

        self.true_label = true_label

        self.setup()

    def setup(self):
        self.path_a = DeepPolyVerifier(self.module.path_a, true_label=self.true_label, DEBUG=self.DEBUG)
        self.path_b = DeepPolyVerifier(self.module.path_b, true_label=self.true_label, DEBUG=self.DEBUG)

        # self.path_a.first_layer.prev = None
        # self.path_b.first_layer.prev = None

        # self.path_a.first_layer.return_constraints = True
        # self.path_b.first_layer.return_constraints = True

        # Used to skip layers in the setup
        for module in list(self.module.path_a):
            self.children.append(module)
        for module in list(self.module.path_b):
            self.children.append(module)

    def forward(self, lb, ub):
        # Analyze the ResNet function

        if self.first_forward:
            # Setup the structure to backpropagate
            self.path_a.first_layer.prev = self.prev
            self.path_b.first_layer.prev = self.prev
            self.first_forward = False

        # Forward of path_a
        lb_a, ub_a = self.path_a.first_layer(lb, ub)

        # Forward of path_b
        lb_b, ub_b = self.path_b.first_layer(lb, ub)

        self.lb = lb_a + lb_b
        self.ub = ub_a + ub_b

        if self.DEBUG:
            # To make sure lb <= ub
            assert not (True in torch.gt(self.lb, self.ub)), "Forward: lb > ub in " + self.name

        back_lb, back_ub, _ = self.backsubstitution()

        # Debug:
        debug(self.lb, self.ub, back_lb, back_ub, self.name)

        # See if we get a tighter bound
        # self.lb = torch.max(back_lb, self.lb)
        # self.ub = torch.min(back_ub, self.ub)
        self.lb = back_lb
        self.ub = back_ub

        if self.DEBUG:
            # To make sure lb <= ub
            assert not (True in torch.gt(self.lb, self.ub)), "lb > ub in " + self.name

        if self.next is not None:
            # Maybe detach lb and ub from computational graph just to be sure
            # self.lb = self.lb.detach()
            # self.ub = self.ub.detach()
            return self.next.forward(self.lb, self.ub)
        else:
            return self.lb, self.ub

    def backsubstitution(self, prev_constraints=None, constraints_only=False):
        # Make sure bias gets counted only once
        if prev_constraints is not None:
            lc_bias = prev_constraints.lc_bias
            uc_bias = prev_constraints.uc_bias

            prev_constraints.lc_bias = torch.zeros(lc_bias.size())
            prev_constraints.uc_bias = torch.zeros(uc_bias.size())

        # Make sure that when we backsubstitude here, path_a goes only back to the first layer
        self.path_a.first_layer.prev = None
        # Backsubstitude only path_a nd store the constraints
        _, _, constraint_a = self.path_a.last_layer.backsubstitution(prev_constraints, constraints_only=True)
        # Make sure that in the forward pass of path_a it will go through the whole network to tighten the constraints
        self.path_a.first_layer.prev = self.prev

        # Do the same for path_b
        self.path_b.first_layer.prev = None
        _, _, constraint_b = self.path_b.last_layer.backsubstitution(prev_constraints, constraints_only=True)
        self.path_b.first_layer.prev = self.prev

        # print("lb_a:", lb_a.size(), "ub_a:", ub_a.size())
        # print("lb_b:", lb_b.size(), "ub_b:", ub_b.size())
        combined_constraint = combine_constraints(constraint_a, constraint_b)
        if prev_constraints is not None:
            combined_constraint.lc_bias += lc_bias
            combined_constraint.uc_bias += uc_bias

            prev_constraints.lc_bias = lc_bias
            prev_constraints.uc_bias = uc_bias

        lb, ub, constraint = self.prev.backsubstitution(combined_constraint, constraints_only=constraints_only)
        return lb, ub, constraint

        # This corresponds to backsubstituting back through the whole graph
        # lb_a, ub_a, c_a = self.path_a.last_layer.backsubstitution(prev_constraints)
        # lb_b, ub_b, c_b = self.path_b.last_layer.backsubstitution(prev_constraints)
        # return lb_a + lb_b, ub_a + ub_b, combine_constraints(c_a, c_b)

    def get_grad_vars(self):
        grad_vars = []
        for layer in self.children:
            if isinstance(layer, ReLUVerifier):
                grad_vars.append(layer.alphas)
            elif isinstance(layer, ResNetVerifier):
                for var in layer.get_grad_vars():
                    grad_vars.append(var)
        return grad_vars


class BatchNormVerifier(Verifier):
    def __init__(self, module, prev=None, next=None):
        super(BatchNormVerifier, self).__init__(prev, next, "BatchNormVerifier")
        self.module = module
        self.mean = module.running_mean.detach()
        self.var = module.running_var.detach()
        self.eps = module.eps
        self.gamma = module.weight.data.detach()
        self.beta = module.bias.data.detach()
        self.weight = None
        self.bias = None

    def setup(self, lb, ub):
        # Calculating weight and bias of this layer:
        divisor = torch.ones(1) / torch.sqrt(self.var + self.eps)
        self.weight = self.gamma * divisor
        self.bias = - self.gamma * self.mean * divisor + self.beta

        # Repeat the vectors to match the size of the flattend minibatch
        numb_batches = self.mean.view(-1).size(0)
        repeats = lb.view(-1).size(0) // numb_batches
        self.weight = self.weight.repeat_interleave(repeats)
        self.bias = self.bias.repeat_interleave(repeats)

        # Diagonal of weight corresponds to elementwise multiplication
        self.weight = torch.diag(self.weight)

        self.constraint = DPConstraints(self.weight, self.weight, self.bias, self.bias)

    def forward(self, lb, ub):
        if self.constraint is None:
            self.setup(lb, ub)

        return super().forward(lb, ub)


class FinalLossVerifier(Verifier):
    """
    Ued as last verifier layer and gives us the loss back
    """
    def __init__(self, true_label, prev=None, next=None):
        super(FinalLossVerifier, self).__init__(prev, next, "FinalLossVerifier")
        self.true_label = true_label
        values = - torch.eye(10)
        values[:, self.true_label] = 1
        self.weight = values.detach()
        self.bias = torch.zeros(10)
        self.constraint = DPConstraints(self.weight, self.weight, self.bias, self.bias)


class DPConstraints:
    def __init__(self, lc, uc, lc_bias, uc_bias):
        # The lower and upper constraints of this layer
        self.lc = lc
        self.uc = uc
        # The lower and upper bias up to this layer (from back to front)
        self.lc_bias = lc_bias
        self.uc_bias = uc_bias
        self.zero = torch.tensor([0.])

    def positive_lc(self):
        return torch.max(self.lc, self.zero)

    def positive_uc(self):
        return torch.max(self.uc, self.zero)

    def negative_lc(self):
        return torch.min(self.lc, self.zero)

    def negative_uc(self):
        return torch.min(self.uc, self.zero)

    def transposed_constraint(self):
        """
        :return: New DPConstraint where lc and uc are transposed
        """
        return DPConstraints(self.lc.T, self.uc.T, self.lc_bias, self.uc_bias)


def combine_constraints(ca, cb):
    """
    :param ca: The first constraint
    :param cb: The second constraint
    :return: DPConstraints that is the combination of cb nd ca
    """
    if ca is None:
        return cb
    elif cb is None:
        return ca
    if isinstance(ca, DPConstraints) and isinstance(cb, DPConstraints):
        return DPConstraints(ca.lc + cb.lc, ca.uc + cb.uc, ca.lc_bias + cb.lc_bias, ca.uc_bias + cb.uc_bias)
    else:
        assert False, "It is only allowed to combine two DPConstraints toghether but got: " + str(ca) + " " + str(
            cb)

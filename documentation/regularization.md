# Regularization

## Contents
##### 1. About overfitting
##### 2. Regularization
##### 3. Parameter norm penalties
##### 4. Lagrange formulation
##### 5. Reprojection and techniques
##### 6. Early stopping
##### 7. Dataset augmentation
##### 8. Bagging and other ensemble methods
##### 9. Dropouts


## What is overfitting?

> When a network tries to mug up from the training data, to deliberately minimize the loss function.

When the model is overtrained over the same training samples, the model fails to generalize to the new unseen data. 

## Overview

![Overfitting](https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Overfitting.svg/320px-Overfitting.svg.png)

In the above image we see the green line seems a forced fit and might underperform on any unseen blue/ black point.

> A good model is a simpler model, which generalized well on limited training data.

Three terms to look for:
1. Variance: It is the model characteristic to change output wrt. change in input. To generalize a model well, variance is possibly lowered.
2. Bias: It is the model characteristic to being biased to an output no matter the change in input. 
3. Complexity: Complexity is the no. of variables model adjusts to fit/ model the data. A model with fewer variables/weights is preferred.

In this graph of loss, we see the training loss is lowering but the validation loss suddently goes upwards. This is due to overfitting.
and the loss is called **generalization loss**

![Loss graph overfitting](https://elitedatascience.com/wp-content/uploads/2017/09/early-stopping-graphic.jpg)

As we increase the model variables/weights to fit the model, it learns or updates unnecessary variables and overfits.
Overfitting models, have high variance and low bias. 

> Our target is to lower the variance significantly while not overly increasing the bias. Here's where regularization comes into picture.

## Regularization

**Modification made to the learning algorithm to reduce generalization error and not train error.**

> It is better to be ignorant to some details of the data, for the model to learn the true behaviour.

The goals of regularizations are:
1. Encode prior knowledge.
2. Prefering for simpler model with fewer parameters.
3. Making undetermined problem determined.

## Parameter Norm penalties.
What is an objective function?
A loss function which the model tries to minimize to fit better on the data.

We add an extra term -' Regularization parameter' to control the complexity of the model.

![Objective function](Image url)

where αε[0,θ) is a hyperparameter that weight the relative contribution of the norm penalty term Ω
Here Theta is the weights or the parameters of the model, X,Y are the features and output of the data.
Alpha is a hyperparameter to tune the effect of regularization. 

### L2 Regularization

When our training algorithm minimizes the regularized objective function.
Norm penalty Ω penalizes only weights at each layer and leaves biases unregularized
It is also called **Weight decay or ridge regression or Tikhonov regularization**

#### Impementation
```
# a short implementation of L2 Norm in pytorch. 
lambda = torch.tensor(1.)
l2_reg = torch.tensor(0.)
for param in model.parameters():
    l2_reg += torch.norm(param)
loss += lambda * l2_reg
```

### L1 Regularization
L1 regularization, another way to penalize model parameters, is defined as **the sum of the absolute values of the individual parameters**

![l1 norm]()

#### Sparsity and feature selection
- The sparsity property induced by L1 regularization has been used extensively as a feature selection mechanism
- LASSO (Least Absolute Shrinkage and Selection Operator) integrates an L1 penalty with a linear model and least squares cost function
- The L1 penalty causes a subset of the weights to become zero, suggesting that those features can be discarded

#### Impementation
```
L1_reg = torch.tensor(0., requires_grad=True)
for name, param in model.named_parameters():
    if 'weight' in name:
        L1_reg = L1_reg + torch.norm(param, 1)

total_loss = total_loss + 10e-4 * L1_reg
```
## Early stopping
## Data augmentation
## Bagging and other ensemble methods
## Dropout
```
torch.nn.Dropout(0.3) # 30% dropping weights
```
## References

https://discuss.pytorch.org/t/how-does-one-implement-weight-regularization-l1-or-l2-manually-without-optimum/7951.
http://pytorch.org/docs/master/torch.html?highlight=norm#torch.norm.

#%matplotlib inline
import numpy as np
import torch
torch.set_printoptions(edgeitems=2)
# 'Celcius' values
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0,
                    3.0, -4.0, 6.0, 13.0, 21.0])

# 'Unknown' values
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4])
t_un = 0.1 * t_u


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    # Q : Why square, don't we want direction?
    # --> A : We get the direction from the derivative of the loss_fn()
    squared_diffs = (t_p - t_c)**2
    # In Python, **2 squares 'element-wise'.  Doing a 'mean' is effectively like
    # doing a dot product w/ a scaling factor of \frac{1}{N}
    # --> Recall from ESL by Hastie, this is effectively the squared error loss
    # --> We want a single scalar value for the slope of the line, so information
    #     from all points need included.
    return squared_diffs.mean()

# Starting guess
params = torch.tensor([1.0, 0.0], requires_grad=True)

print(params.grad is None)

# Let's analytically calculate for t_c[0], t_u[0]
#   A simple linear model takes 'unknown' input and returns 'predicted' values :
#       $$
#           m(t_{u}, w, b) & = w \times t_{u} + b
#                          & = t_{p}
#       $$
#
#   Loss function takes the difference between the 'known' celcius values and the
#   'predicted' values from the model.
#
#       $$
#           params = [1.0, 0.0]
#
#           t_{p} & = m(t_{u}, *params)                                     \\
#                 & = m(t_{u}, w=params[0], b=params[1])                    \\
#                 & = w \times t_{u} + b                                    \\
#                                                                           \\
#                 \text{NOTE : $t_{u}$ is a vector}                         \\
#                 & = params[0] \times t_{u} + params[1]                    \\
#                                                                           \\
#                                                                           \\
#           loss_fn(t_{p}, t_{c}) & =                                       \\
#                 \text{Expanding 'mean' to summation below}                \\
#                 & = \frac{\sum_{i}^{N} (t_{pi} - t_{ci})^{2}}{N}          \\
#                                                                           \\
#                 \text{Substitute 'predicted' values from above}           \\
#                 & = \frac{\sum_{i}^{N} ((w \times t_{ui} + b) - t_{ci})^{2}}{N}  \\
#                 & = \frac{\sum_{i}^{N} ((params[0] \times t_{ui} + params[1]) - t_{ci})^{2}}{N} \\
#       $$
#
#
#   Compute Derivative w/r/t $w$
#       $$
#           \frac{d loss_fn}{dw} \\
#                 \text{Chain Rule}                                                \\
#                 & = \frac{d loss_fn()}{d t_{pi}} \frac{d t_{pi}}{dw}             \\
#                                                                                  \\
#                 & = \frac{ d \frac{\sum_{i}^{N} ((t_{pi} - t_{ci})^{2})}{N}}{d t_{pi}} \frac{d t_{pi}}{dw}  \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \frac{t_{pi}}{dw} \\
#                                                                                  \\
#                 \text{Compute $\frac{t_{pi}}{dw}$}                               \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \frac{d (w \times t_{ui} + b)}{dw}  \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \times t_{ui}     \\
#                                                                                  \\
#                 \text{Subsitute for t_{pi}}                                      \\
#                 & = \frac{\sum_{i}^{N} 2 ((w \times t_{ui} + b) - t_{ci})}{N} \times t_{ui}     \\
#                                                                                  \\
#                 \text{Let's collapse this back to the Python code notation}      \\
#                 & = 2 * ((w * t_u + b - t_c) * t_u).mean()                  \\
#                 & = 2 * ((params[0] * t_u + params[1] - t_c) * t_u).mean()                  \\
#       $$
#
#
#   Compute Derivative w/r/t $b$
#       $$
#           \frac{d loss_fn}{db} \\
#                 \text{Chain Rule}                                                \\
#                 & = \frac{d loss_fn()}{d t_{pi}} \frac{d t_{pi}}{db}             \\
#                 & = \frac{ d \frac{\sum_{i}^{N} ((t_{pi} - t_{ci})^{2})}{N}}{d t_{pi}} \frac{d t_{pi}}{db}  \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \frac{t_{pi}}{db} \\
#                                                                                  \\
#                 \text{Compute $\frac{t_{pi}}{db}$}                               \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \frac{d (w \times t_{ui} + b)}{db}  \\
#                 & = \frac{\sum_{i}^{N} 2 (t_{pi} - t_{ci})}{N} \times 1          \\
#                 & = \frac{\sum_{i}^{N} 2 ((w \times t_{ui} + b) - t_{ci})}{N}    \\
#                                                                                  \\
#                 \text{Let's collapse this back to the Python code notation}      \\
#                 & = 2 * (w * t_u + b - t_c).mean()
#                 & = 2 * (params[0] * t_u + params[1] - t_c).mean()
#       $$
#
#
#
# *params causes loss_fn to be called on each element in params tensor
loss = loss_fn(model(t_u, *params), t_c)
# --> Equivalent to ((t_c - model(t_u, params[0], params[1]))**2).mean()
loss.backward()
# --> From above : derivative w/r/t $w$
#   $$
#     \frac{d loss_fn}{dw} & = 2 \times ((w \times t_{u} + b) - t_{c}) \times t_{u}
#                          & = 2 * ((params[0] * t_u + params[1]) - t_c) * t_u
#                          & = ( 2 * ((params[0] * t_u + params[1]) - t_c) * t_u).mean()
#                          & = tensor(4517.2964, grad_fn=<MeanBackward0>)
#   $$
#
# --> From above : derivative w/r/t $b$
#   $$
#     \frac{d loss_fn}{dw} & = 2 \times ((w \times t_{u} + b) - t_{c}) \times 1
#                          & = 2 * ( (params[0] * t_u + params[1]) - t_c).mean()
#                          & = ((2 * ((params[0] * t_u + params[1]) - t_c))).mean()
#                          & = tensor(82.6000, grad_fn=<MulBackward0>)
#   $$
#
# Now :
#   (Pdb) p params.grad
#   tensor([4517.2969,   82.6000])
# --> It matches!
params.grad


if params.grad is not None:
    params.grad.zero_()


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>\n",
            params.grad.zero_()
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        with torch.no_grad():  # <2>\n",
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params

training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True), # <1>
    t_u = t_un, # <2>
    t_c = t_c)

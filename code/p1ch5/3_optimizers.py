# Purpose :
#   Here we have two thermometers, one in known values (in Celcius) and the other
#   in unknown units.  It is important to note that the data t_c and t_u
#   are collected at the same points in time.
#
#   So basically, this model is providing the slope and offset of the 'unknown'
#   values.
#
#   Output of this code converges to : params = tensor([  5.3671, -17.3012])
#       Equation for converting Fareinheit to Celcius is
#        $$
#           T_{C}   &= (T_{F} - 32) \times \frac{5}{9} \\
#                   &= (T_{F} - 32) \times \frac{5}{9} \\
#                   &= \frac{5}{9} T_{F} - 17.7 \\
#        $$
#           #. Note that our final params are $[w,b] = [5.3671, -17.3012]$.
#           #. We need to divide our $w$ by 10 and we get close to $\frac{5}{9}$
#           #. w = 5/9 * 10 = 5.5
#
#
import torch
import numpy as np
import torch.optim as optim
torch.set_printoptions(edgeitems=2, linewidth=75)

# 'Celcius' values
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0,
                    8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

# 'Unknown' values
t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
                    33.9, 21.8, 48.4, 60.4, 68.4])

# p120 - this is a cheap and easy way to have the values stay close to [-1,1]
#   --> e.g. think regularize / normalize
t_un = 0.1 * t_u

## Define Model
def model(t_u, w, b):
    return w * t_u + b

## Define Loss function
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

print(f'Available Optimizers:')
for item in dir(optim):
    print(f'\t{item}')

###########
###########
# This section illustrates that optimizer updates params w/o
# having to manipulate it by hand, like we did in 2_autograd.py
# recall :
#       with torch.no_grad():
#           params -= learning_rate * params.grad
print(f'\n\nUsing raw unknown data (t_u) w/ single update from optimzer: ')
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
print(f'    params = {params}')
print(f'    learning_rate = {learning_rate}')
print(f'    optimizer = optimizer.SGD')

# SGD = Stochastic Gradient Descent
#   --> if momentum = 0 [default], it is vanilla gradient descent
#   --> 'stochastic' part comes from averaging over random subsets of all input
#       samples, called \emph{minibatch}
#   --> Algorithm does same thing, regardless of whether 'vanilla' or 'stochastic'
#       version run
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_u, *params)

loss = loss_fn(t_p, t_c)
loss.backward()

# See : updates params w/o 'torch.no_grad()'
optimizer.step()

print(f'    (post-update using t_u);\n        params = {params}')
###########
###########


###########
###########
# Redo same experiment with 'normalized' unknown data and different learning_rate
#  --> Don't forget to zero out the gradients for optimizer
#  --> This chunk, would be 'loop' ready
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
print(f'\n\nNow using "normalized" unknown data (t_un), reset params/lr : ')
print(f'    params = {params}')
print(f'    learning_rate = {learning_rate}')
print(f'    optimizer = optimizer.SGD')
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

# Exact placement of zero_grad() is somewhat arbitrary
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'    (post-update using t_un);\n        params = {params}')
print(f'\n\n')
###########
###########




###########
###########
print(f'Now do a training loop with optim.SGD and pseudo-normalized unknown data:')
# Now we can abstract away the specific optimization scheme
def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        # p129 - Recall we need to reset gradient, this is just how
        #        derivatives accumulate in pytorch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print('    Epoch %d, Loss %f' % (epoch, float(loss)))

    return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
print(f'    params = {params}')
print(f'    learning_rate = {learning_rate}')
print(f'    optimizer = optimizer.SGD')
## IMPORTANT - Crutial that 'params' used to initilize the optimizer is the
##             same as passed below in training loop
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    ## IMPORTANT - Crutial this is same 'params' used to initilize the optimizer
    ##             b/c this params is used by the model and loss_fn where the
    ##             gradient gets computed.
    params = params,
    ## use pseudo-normalized unknown data
    t_u = t_un,
    t_c = t_c)


print(f'    params = {params}')
print(f'\n\n')
###########
###########



###########
###########
# Try with different optimizer with more aggressive learning rate :
# About Adam optimizer :
#    --> more sophisticated optimizer
#    --> which the learning rate is set adaptively
#    --> It is a lot less senistive to scaling of the parameters
print(f'Now do a training loop with optim.Adam and RAW unknown data:')
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # <1>
print(f'    params = {params}')
print(f'    learning_rate = {learning_rate}')
print(f'    optimizer = optim.Adam')

training_loop(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    ## Agressive...using non-normalized data b/c Adam optimizer doesn't care
    t_u = t_u, # <2>
    t_c = t_c)

print(f'    params = {params}')
print(f'\n\n')
###########
###########



########## 5.5.3 Training, validation, and overfitting ##########
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices  # <1>


train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u,
                  train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params) # <1>
        train_loss = loss_fn(train_t_p, train_t_c)

        val_t_p = model(val_t_u, *params) # <1>
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward() # <2>
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f'Epoch {epoch}, Training loss {train_loss.item():.4f},'
                  f'Validation loss {val_loss.item():.4f}')

    return params

params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    params = params,
    train_t_u = train_t_un, # <1>
    val_t_u = val_t_un, # <1>
    train_t_c = train_t_c,
    val_t_c = val_t_c)

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u,
                  train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad(): # <1>
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
            assert val_loss.requires_grad == False # <2>

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss

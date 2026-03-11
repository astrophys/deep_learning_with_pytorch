#matplotlib inline,
import numpy as np
import torch
torch.set_printoptions(edgeitems=2, linewidth=75)
# Celcius
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
# Unknown units
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# Confusing that they use t_u b/c the model could be t
# --> This is a linear model
def model(t_u, w, b):
    return w * t_u + b

# Mean square loss
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
t_p

loss = loss_fn(t_p, t_c)
loss
x = torch.ones(())
y = torch.ones(3,1)
z = torch.ones(1,3)
a = torch.ones(2, 1, 1)
print(f"shapes: x: {x.shape}, y: {y.shape}")
print(f"        z: {z.shape}, a: {a.shape}")
print("x * y:", (x * y).shape)
print("y * z:", (y * z).shape)
print("y * z * a:", (y * z * a).shape)

delta = 0.1

# Here we are comparing the uncertain temps to the Truth data (t_c)
# --> They compare model to truth, this makes sense
loss_rate_of_change_w = ((loss_fn(model(t_u, w + delta, b), t_c) -
                         loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta))
learning_rate = 1e-2

# Change weights based off gradient times learning rate
w = w - learning_rate * loss_rate_of_change_w

loss_rate_of_change_b = ((loss_fn(model(t_u, w, b + delta), t_c) -
                          loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta))

b = b - learning_rate * loss_rate_of_change_b

### It is weird having functions take arguments that they don't use...

# derivative = \frac{d loss_fun}{d t_p}
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)  # <1>
    return dsq_diffs

# derivative = \frac{d model}{dw}
def dmodel_dw(t_u, w, b):
    return t_u

# derivative = \frac{d model}{db}
def dmodel_db(t_u, w, b):
    return 1.0

# This is a vector
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])  # <1>


def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    """

    Args :
        n_epochs      = number of iterations for training loop
        learning_rate = float, less than 1
        params        = tuple, w (weights), b (offset)
        t_u           = temperature in unknown units
        t_c           = truth value in celcius

    Returns :
    Raises :
    """
    for epoch in range(1, n_epochs + 1):
        w, b = params

        # Forward pass
        t_p = model(t_u, w, b)  # <1>
        # Question : What is the point of the below line? grad_fun calls loss_fn
        #            directly?
        loss = loss_fn(t_p, t_c)
        # Backward pass
        grad = grad_fn(t_u, t_c, t_p, w, b)  # <2>

        params = params - learning_rate * grad

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss {loss:.4f}, grad(w,b) = ({grad[0]:.4f}, {grad[1]:.4f}), params (w,b) = ({params[0]:.4f}, {params[1]:.4f}') # <3>

    return params



### This loop does NOT converge
# LOOP 1
print(f'\n\nLoop 1')
print(f'n_epochs = 100, learning_rate = 1e-2')
training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

# LOOP 2
print(f'\n\nLoop 2')
print(f'n_epochs = 100, learning_rate = 1e-4')
training_loop(
    n_epochs = 100,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)

# LOOP 3
print(f'\n\nLoop 3')
print(f'n_epochs = 100, learning_rate = 1e-2')
# Section 5.4.4
#   --> Not normalizing, but just scaling down
#   --> Note that learning rate is 100x larger than Loop 2, with similar results
t_un = 0.1 * t_u
training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un, # <1>
    t_c = t_c)

# Ali commented b/c it is defined above...
def training_loop(n_epochs, learning_rate, params, t_u, t_c,
                  print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)  # <1>
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # <2>

        params = params - learning_rate * grad

        if epoch in {1, 2, 3, 10, 11, 99, 100, 4000, 5000}:  # <3>
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            print('    Params:', params)
            print('    Grad:  ', grad)
        if epoch in {4, 12, 101}:
            print('...')

        if not torch.isfinite(loss).all():
            break  # <3>

    return params

# LOOP 4
print(f'\n\nLoop 4')
print(f'n_epochs = 5000, learning_rate = 1e-2')
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c,
    print_params = False)

#%matplotlib inline
from matplotlib import pyplot as plt
print(f'Final Params = {params}')

t_p = model(t_un, *params)  # <1>

fig = plt.figure(dpi=600)
plt.xlabel('Temperature (°Fahrenheit)')
plt.ylabel('Temperature (°Celsius)')
plt.plot(t_u.numpy(), t_p.detach().numpy()) # <2>
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig('fig_5.9_unknown_plot.png', format='png')  # bookskip


fig = plt.figure(dpi=600)
plt.xlabel('Measurement')
plt.ylabel('Temperature (°Celsius)')
plt.plot(t_u.numpy(), t_c.numpy(), 'o')

plt.savefig('fig_5.9_data_plot.png', format='png')

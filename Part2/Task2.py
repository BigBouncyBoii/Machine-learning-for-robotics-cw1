# CODE MAP of Task2.py
#   Settings       learning rate, number of epochs, random seed, ADAM constants
#   Load data      read RegressionData.csv, make xtest, reset_weights() loads InitialWeights.npz
#   Forward pass   hidden outputs a1 and network output y (one sample or many), training error
#   Deltas         output delta, then hidden delta, then the gradient of every weight and bias
#   Update         SGD: new weights and biases after every sample, training loop
#   ADAM           same network trained with ADAM: m, v, t, bias correction, training loop
#   Run            train with SGD, then with ADAM, each from the given weights
#   Plots          fit on xtest, both learning curves on one figure, printed errors
#   AI use         Claude Opus 5.5 (claude-opus-5-5), default setting
#
# HOW TO RUN (from the folder holding this file and the data files)
#   python3 -m venv .venv
#   source .venv/bin/activate        (Windows: .venv\Scripts\activate)
#   pip install numpy pandas matplotlib
#   python Task2.py
#
# THE BIG PICTURE
#   The network is 1-3-1: 1 input x -> 3 hidden nodes (tanh) -> 1 output y (linear).
#   Goal: learn the curve d(x) from 41 points, so y is close to d.
#   Each training step does 3 things:
#     1. Forward pass:  put x in, get the output y.
#     2. Backward pass: work out how much each weight caused the error (the gradient).
#     3. Update:        move each weight a small step to make the error smaller.
#   We train twice from the same starting weights: once with SGD, once with ADAM.

# %% Settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ETA = 0.1            # SGD learning rate: how big each step is
N_EPOCHS = 3000      # 1 epoch = one pass through all 41 training points
SEED = 67            # fixes the random shuffle order, so every run gives the same result
ERR_TARGET = 0.001   # we report the first epoch where the error drops below this

# ADAM settings (the usual default values, except the step size)
ALPHA = 0.005        # ADAM step size (plays the same role as ETA)
BETA1 = 0.9          # how much ADAM remembers past gradients (the average direction)
BETA2 = 0.999        # how much ADAM remembers past squared gradients (the average size)
EPS = 1e-8           # tiny number so we never divide by zero

OUT_DIR = "results_normal"           # folder where the figures are saved
os.makedirs(OUT_DIR, exist_ok=True)

# %% Load data
df = pd.read_csv("RegressionData.csv")
x, d = df["x"].values, df["d"].values          # x = inputs, d = desired outputs (41 each)

xtest = np.linspace(-0.97, 0.93, 20)           # 20 new x values, used only to plot the trained curve


def reset_weights():
  """Set W1, b1, W2, b2 back to the given starting weights."""
  # Both SGD and ADAM start from these same weights, so the comparison is fair.
  global W1, b1, W2, b2
  w = np.load("InitialWeights.npz")
  W1, b1 = w["W1r"], w["b1r"]                  # input -> hidden: 3 weights, 3 biases
  W2, b2 = w["W2r"], w["b2r"]                  # hidden -> output: 3 weights, 1 bias


reset_weights()

# %% Forward pass
def forward(xin):
  """xin: scalar or 1-D array of N inputs. Returns a1 (3,N) and y (1,N)."""
  xin = np.reshape(xin, (1, -1))               # make x a row, so one x or many x both work
  a1 = np.tanh(W1 @ xin + b1)                  # hidden layer: weighted sum n1, then tanh
  y = W2 @ a1 + b2                             # output layer: weighted sum n2, no activation (y = n2)
  return a1, y                                 # a1 is kept because backprop needs it


def training_error():
  """Mean of (d - y)^2 over all 41 training samples."""
  # Mean squared error: average of (target - output)^2. Lower is better.
  _, y = forward(x)
  return np.mean((d - y.ravel()) ** 2)


# %% Deltas
# A "delta" is how much the error changes if a node's weighted sum (n) changes.
# We find the output delta first, then pass it backwards to the hidden layer.
# This is the chain rule, which is why it's called backpropagation.
def deltas(di, a1, y):
  """Return delta1 (3,1) for the hidden nodes and delta2 (1,1) for the output node."""
  # --- Output node ---
  # Error for one sample: E = ½(d - y)².  The ½ cancels the 2 when we differentiate.
  dEdy = -(di - y)                 # dE/dy: how the error changes with the output
  dydn2 = 1                        # output is linear (y = n2), so its slope is 1
  delta2 = dEdy * dydn2            # output delta = (y - d)

  # --- Hidden nodes ---
  # Chain rule: delta1 = delta2 x (weight from this hidden node to output) x (tanh slope)
  dn2da1 = W2.T                    # each hidden node affects the output through its weight W2
  da1dn1 = 1 - a1**2               # slope of tanh is 1 - tanh², and tanh(n1) is just a1
  delta1 = delta2 * dn2da1 * da1dn1   # one delta for each of the 3 hidden nodes
  return delta1, delta2


def gradients(xi, di, a1, y):
  """Return dE/dW1, dE/db1, dE/dW2, dE/db2 for one sample (same shapes as the weights)."""
  delta1, delta2 = deltas(di, a1, y)
  # Rule: gradient of a weight = (delta of the node it goes into) x (the value coming into it)

  # Hidden -> output weights: the value coming in is the hidden output a1
  dn2dw2 = a1.T
  dEdw2 = delta2 * dn2dw2
  dEdb2 = delta2                 # a bias is like a weight whose input is always 1

  # Input -> hidden weights: the value coming in is the input x
  dn1dw1 = xi
  dEdw1 = delta1 * dn1dw1
  dEdb1 = delta1                 # again, bias input is 1
  return dEdw1, dEdb1, dEdw2, dEdb2


# %% Update
def update_sgd(xi, di, a1, y):
  """One SGD step: w <- w - eta * dE/dw for every weight and bias."""
  # The gradient points uphill (more error), so we step the opposite way (minus sign).
  global W1, b1, W2, b2
  dEdw1, dEdb1, dEdw2, dEdb2 = gradients(xi, di, a1, y)
  # All gradients are worked out first (above), using the old weights.
  # Only then do we change the weights.
  W2 = W2 - ETA * dEdw2
  b2 = b2 - ETA * dEdb2
  W1 = W1 - ETA * dEdw1
  b1 = b1 - ETA * dEdb1


def train_sgd():
  """Train from the given weights with SGD. Returns the training error after each epoch."""
  reset_weights()                # start from the given weights
  np.random.seed(SEED)           # same shuffle order every time we run
  err = np.zeros(N_EPOCHS)       # stores the error after each epoch (for the learning curve)
  for epoch in range(N_EPOCHS):
    # "Stochastic": update after EVERY single sample, not after the whole dataset.
    # Shuffling each epoch stops the network learning the points in a fixed order.
    for i in np.random.permutation(len(x)):
      a1, y = forward(x[i])                      # 1. forward pass for this one point
      update_sgd(x[i], d[i], a1, y)              # 2+3. backprop and update the weights
    err[epoch] = training_error()                # error on all 41 points after this epoch
  return err


# %% ADAM
# ADAM is SGD with two improvements:
#   m = running average of the gradient       -> smooths out noisy steps (like momentum)
#   v = running average of the gradient²      -> measures how big the gradients usually are
# Step = ALPHA x m / sqrt(v), so each weight gets its own step size:
#   weights with big gradients take smaller steps, weights with small gradients take bigger steps.
def update_adam(xi, di, a1, y, m, v, t):
  """One ADAM step. m and v are lists of 4 arrays (one per weight/bias), updated in place; t is the step count."""
  global W1, b1, W2, b2
  grads = gradients(xi, di, a1, y)     # same gradients as SGD; only the update rule is different
  params = [W1, b1, W2, b2]

  for k in range(4):                   # do the same thing for W1, b1, W2 and b2
    g = grads[k]
    m[k] = BETA1 * m[k] + (1 - BETA1) * g          # update the average gradient
    v[k] = BETA2 * v[k] + (1 - BETA2) * g**2       # update the average squared gradient
    # Bias correction: m and v start at 0, so early on they are too small.
    # Dividing by (1 - beta^t) fixes this. As t grows, this divisor goes to 1 and stops mattering.
    m_hat = m[k] / (1 - BETA1**t)
    v_hat = v[k] / (1 - BETA2**t)
    params[k] = params[k] - ALPHA * m_hat / (np.sqrt(v_hat) + EPS)   # the ADAM step
  W1, b1, W2, b2 = params


def train_adam():
  """Train from the given weights with ADAM. Returns the training error after each epoch."""
  reset_weights()                # same starting weights as SGD (fair comparison)
  np.random.seed(SEED)           # same shuffle order as SGD
  m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]   # m and v start at zero, one per weight array
  v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
  t = 0                          # step counter, used in the bias correction
  err = np.zeros(N_EPOCHS)
  for epoch in range(N_EPOCHS):
    for i in np.random.permutation(len(x)):      # shuffle every epoch, as in SGD
      a1, y = forward(x[i])
      t += 1                                     # counts every update, and is never reset between epochs
      update_adam(x[i], d[i], a1, y, m, v, t)
    err[epoch] = training_error()
  return err


# %% Run
err_sgd = train_sgd()
_, y_test_sgd = forward(xtest)    # save the SGD curve now, because train_adam() resets the weights

err_adam = train_adam()
_, y_test_adam = forward(xtest)   # the ADAM curve

# %% Plots
def first_below(errors, target):
  """Return the first epoch where the error is below target (epochs counted from 1)."""
  below = np.nonzero(errors < target)[0]
  return f"epoch {below[0] + 1}" if below.size else "not reached"


print(f"SGD  final training error: {err_sgd[-1]:.6f}")
print(f"ADAM final training error: {err_adam[-1]:.6f}")
print(f"SGD  first below {ERR_TARGET}: {first_below(err_sgd, ERR_TARGET)}")
print(f"ADAM first below {ERR_TARGET}: {first_below(err_adam, ERR_TARGET)}")

# Figure 1: training points and both trained curves on xtest (shows how well the network fits)
plt.figure()
plt.plot(x, d, "o", label="Training data")
plt.plot(xtest, y_test_sgd.ravel(), "x-", label="SGD output on xtest")
plt.plot(xtest, y_test_adam.ravel(), "+--", label="ADAM output on xtest")
plt.xlabel("x")
plt.ylabel("d, y")
plt.title("1-3-1 network fit")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "Task2_fit.png"), dpi=150, bbox_inches="tight")

# Figure 2: error vs epoch for both methods (shows how fast each one learns)
# The log scale on the y-axis makes small errors visible.
plt.figure()
epochs = np.arange(1, N_EPOCHS + 1)
plt.semilogy(epochs, err_sgd, label="SGD")
plt.semilogy(epochs, err_adam, label="ADAM")
plt.axhline(ERR_TARGET, color="grey", linestyle="--", label=f"{ERR_TARGET}")
plt.xlabel("Epoch")
plt.ylabel("Training error (mean of (d - y)$^2$)")
plt.title("Learning curves")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "Task2_learning_curves.png"), dpi=150, bbox_inches="tight")

plt.show()

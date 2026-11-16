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
# HOW TO RUN (from any folder: data files are found next to this file)
#   python3 -m venv .venv
#   source .venv/bin/activate        (Windows: .venv\Scripts\activate)
#   pip install numpy pandas matplotlib
#   python Task2.py
#
# THE BIG PICTURE
#   1-3-1 network: input x -> 3 tanh hidden nodes -> 1 linear output y. Learn d(x) from 41 points.
#   Each step: forward pass (x -> y), backward pass (gradient of E for every weight), update (step downhill).
#   Trained twice from the same starting weights: SGD, then ADAM.

# %% Settings
import numpy as np                 # arrays and maths (matrix product @, tanh, mean)
import pandas as pd                # only used to read the CSV file
import matplotlib.pyplot as plt    # the two figures
import os                          # builds file paths

HERE = os.path.dirname(os.path.abspath(__file__))   # folder of this file, so paths work from any folder

ETA = 0.1            # SGD learning rate: weight change = ETA x gradient
N_EPOCHS = 3000      # epochs; 1 epoch = one pass through all 41 points
SEED = 67            # seed for the shuffle, so every run gives the same numbers
ERR_TARGET = 0.001   # we report the first epoch with error below this

# ADAM settings (standard defaults, except the step size)
ALPHA = 0.005        # ADAM step size (the ADAM version of ETA)
BETA1 = 0.9          # decay of m, the running average of the gradient
BETA2 = 0.999        # decay of v, the running average of the squared gradient
EPS = 1e-8           # added to sqrt(v_hat) so we never divide by zero

OUT_DIR = "results_normal"              # name of the folder for the saved figures
OUT_DIR = os.path.join(HERE, OUT_DIR)   # make it a full path next to this file
os.makedirs(OUT_DIR, exist_ok=True)     # create it; no error if it already exists

# %% Load data
df = pd.read_csv(os.path.join(HERE, "RegressionData.csv"))   # table with columns x and d
x, d = df["x"].values, df["d"].values          # NumPy arrays: x = inputs (41,), d = targets (41,)

xtest = np.linspace(-0.97, 0.93, 20)           # (20,) evenly spaced new x values, only for plotting the fit


def reset_weights():
  """Set W1, b1, W2, b2 back to the given starting weights."""
  # Called at the start of SGD and of ADAM, so both start from the same weights (fair comparison).
  global W1, b1, W2, b2                        # change the module-level weights, not local copies
  w = np.load(os.path.join(HERE, "InitialWeights.npz"))   # dict-like object of the 4 saved arrays
  W1, b1 = w["W1r"], w["b1r"]                  # input -> hidden: W1 (3x1), b1 (3x1)
  W2, b2 = w["W2r"], w["b2r"]                  # hidden -> output: W2 (1x3), b2 (1x1)


reset_weights()                                # load weights once, so forward() works straight away

# %% Forward pass
def forward(xin):
  """xin: scalar or 1-D array of N inputs. Returns a1 (3,N) and y (1,N)."""
  xin = np.reshape(xin, (1, -1))               # (1xN) row: one x or many x go through the same code
  a1 = np.tanh(W1 @ xin + b1)                  # (3xN) hidden outputs: n1 = W1 x + b1, then tanh (b1 added to every column)
  y = W2 @ a1 + b2                             # (1xN) output: n2 = W2 a1 + b2, linear so y = n2
  return a1, y                                 # a1 returned too: backprop needs it (tanh slope, dE/dW2)


def training_error():
  """Mean of (d - y)^2 over all 41 training samples."""
  _, y = forward(x)                     # run all 41 x at once; y (1x41), a1 not needed
  return np.mean((d - y.ravel()) ** 2)  # ravel: (1x41) -> (41,) to match d; mean squared error, a scalar


# %% Deltas
# delta = dE/dn: how the error changes with a node's weighted sum n.
# Output delta first, then passed back to the hidden nodes (chain rule = backpropagation).
def deltas(di, a1, y):
  """Return delta1 (3,1) for the hidden nodes and delta2 (1,1) for the output node."""
  # --- Output node --- E = ½(d - y)², so dE/dy = -(d - y) (the ½ cancels the 2)
  dEdy = -(di - y)                 # (1x1) dE/dy for this sample (di is one target value)
  dydn2 = 1                        # dy/dn2 = 1, because the output is linear (y = n2)
  delta2 = dEdy * dydn2            # (1x1) output delta = y - d

  # --- Hidden nodes --- delta1 = delta2 x W2 x tanh slope
  dn2da1 = W2.T                    # (3x1) dn2/da1 = W2, transposed to a column (one per hidden node)
  da1dn1 = 1 - a1**2               # (3x1) tanh slope: 1 - tanh(n1)², and tanh(n1) = a1
  delta1 = delta2 * dn2da1 * da1dn1   # (3x1) element-wise product: one delta per hidden node
  return delta1, delta2


def gradients(xi, di, a1, y):
  """Return dE/dW1, dE/db1, dE/dW2, dE/db2 for one sample (same shapes as the weights)."""
  delta1, delta2 = deltas(di, a1, y)   # deltas of the hidden nodes (3x1) and the output node (1x1)
  # Rule: dE/dw = (delta of the node the weight goes into) x (the input that weight multiplies)

  # Hidden -> output: the inputs to the output node are a1
  dn2dw2 = a1.T                  # (1x3) dn2/dW2 = a1, transposed to a row like W2
  dEdw2 = delta2 * dn2dw2        # (1x3) dE/dW2, same shape as W2
  dEdb2 = delta2                 # (1x1) dE/db2 = delta2 (a bias's input is always 1)

  # Input -> hidden: the input to each hidden node is x
  dn1dw1 = xi                    # dn1/dW1 = x, one number
  dEdw1 = delta1 * dn1dw1        # (3x1) dE/dW1, same shape as W1
  dEdb1 = delta1                 # (3x1) dE/db1 = delta1 (bias input is 1)
  return dEdw1, dEdb1, dEdw2, dEdb2


# %% Update
def update_sgd(xi, di, a1, y):
  """One SGD step: w <- w - eta * dE/dw for every weight and bias."""
  global W1, b1, W2, b2                                 # change the module-level weights
  dEdw1, dEdb1, dEdw2, dEdb2 = gradients(xi, di, a1, y)   # all 4 gradients, computed with the OLD weights
  # Minus sign: the gradient points uphill (more error), so we step the other way.
  W2 = W2 - ETA * dEdw2          # (1x3) new hidden -> output weights
  b2 = b2 - ETA * dEdb2          # (1x1) new output bias
  W1 = W1 - ETA * dEdw1          # (3x1) new input -> hidden weights
  b1 = b1 - ETA * dEdb1          # (3x1) new hidden biases


def train_sgd():
  """Train from the given weights with SGD. Returns the training error after each epoch."""
  reset_weights()                # start from InitialWeights.npz
  np.random.seed(SEED)           # same shuffle order on every run
  err = np.zeros(N_EPOCHS)       # (3000,) error after each epoch, for the learning curve
  for epoch in range(N_EPOCHS):  # 3000 passes through the data
    for i in np.random.permutation(len(x)):    # indices 0..40 in a new random order each epoch
      a1, y = forward(x[i])                      # forward pass for one point: a1 (3x1), y (1x1)
      update_sgd(x[i], d[i], a1, y)              # backprop + update after EVERY point (= "stochastic")
    err[epoch] = training_error()                # MSE on all 41 points at the end of the epoch
  return err                                     # (3000,) the SGD learning curve


# %% ADAM
# m = running average of the gradient (direction, smooths noise like momentum)
# v = running average of the gradient² (typical size)
# Step = ALPHA x m / sqrt(v): each weight gets its own step size, roughly ALPHA whatever the gradient size.
def update_adam(xi, di, a1, y, m, v, t):
  """One ADAM step. m and v are lists of 4 arrays (one per weight/bias), updated in place; t is the step count."""
  global W1, b1, W2, b2
  grads = gradients(xi, di, a1, y)     # same 4 gradients as SGD; only the update rule differs
  params = [W1, b1, W2, b2]            # list of the 4 arrays, so one loop handles them all

  for k in range(4):                   # k = 0..3: W1, b1, W2, b2
    g = grads[k]                                   # gradient of this array (same shape as it)
    m[k] = BETA1 * m[k] + (1 - BETA1) * g          # m = 0.9 m + 0.1 g: update average gradient
    v[k] = BETA2 * v[k] + (1 - BETA2) * g**2       # v = 0.999 v + 0.001 g²: update average squared gradient
    m_hat = m[k] / (1 - BETA1**t)                  # bias correction: m starts at 0 so is too small early on
    v_hat = v[k] / (1 - BETA2**t)                  # same for v; both divisors -> 1 as t grows
    params[k] = params[k] - ALPHA * m_hat / (np.sqrt(v_hat) + EPS)   # the ADAM step for this array
  W1, b1, W2, b2 = params              # copy the new arrays back into the module-level weights


def train_adam():
  """Train from the given weights with ADAM. Returns the training error after each epoch."""
  reset_weights()                # same starting weights as SGD (fair comparison)
  np.random.seed(SEED)           # same shuffle order as SGD
  m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]   # m starts at 0: one zero array per weight array
  v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]   # v starts at 0, same shapes
  t = 0                          # update counter for the bias correction
  err = np.zeros(N_EPOCHS)       # (3000,) error after each epoch
  for epoch in range(N_EPOCHS):
    for i in np.random.permutation(len(x)):      # shuffle every epoch, as in SGD
      a1, y = forward(x[i])                      # forward pass for one point
      t += 1                                     # one more update; never reset between epochs
      update_adam(x[i], d[i], a1, y, m, v, t)    # backprop + ADAM update (changes m, v and the weights)
    err[epoch] = training_error()                # MSE on all 41 points
  return err                                     # (3000,) the ADAM learning curve


# %% Run
err_sgd = train_sgd()             # train with SGD -> (3000,) errors
_, y_test_sgd = forward(xtest)    # (1x20) SGD output on xtest; saved now because train_adam() resets the weights

err_adam = train_adam()           # train with ADAM from the same start -> (3000,) errors
_, y_test_adam = forward(xtest)   # (1x20) ADAM output on xtest

# %% Plots
def first_below(errors, target):
  """Return the first epoch where the error is below target (epochs counted from 1)."""
  below = np.nonzero(errors < target)[0]   # indices (0-based) of epochs with error < target
  return f"epoch {below[0] + 1}" if below.size else "not reached"   # first one, +1 so epochs count from 1


print(f"SGD  final training error: {err_sgd[-1]:.6f}")    # error after the last epoch, 6 decimals
print(f"ADAM final training error: {err_adam[-1]:.6f}")
print(f"SGD  first below {ERR_TARGET}: {first_below(err_sgd, ERR_TARGET)}")
print(f"ADAM first below {ERR_TARGET}: {first_below(err_adam, ERR_TARGET)}")

# Figure 1: the fit. Training points and both trained curves on xtest.
plt.figure()                                                         # new figure window
plt.plot(x, d, "o", label="Training data")                           # 41 data points as circles
plt.plot(xtest, y_test_sgd.ravel(), "x-", label="SGD output on xtest")    # ravel: (1x20) -> (20,) to plot
plt.plot(xtest, y_test_adam.ravel(), "+--", label="ADAM output on xtest")
plt.xlabel("x")
plt.ylabel("d, y")
plt.title("1-3-1 network fit")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "Task2_fit.png"), dpi=150, bbox_inches="tight")   # save to results_normal/

# Figure 2: learning curves. Error vs epoch for both methods.
plt.figure()
epochs = np.arange(1, N_EPOCHS + 1)       # (3000,) epoch numbers 1..3000 for the x-axis
plt.semilogy(epochs, err_sgd, label="SGD")     # log y-axis, so small errors near 0.001 are visible
plt.semilogy(epochs, err_adam, label="ADAM")
plt.axhline(ERR_TARGET, color="grey", linestyle="--", label=f"{ERR_TARGET}")   # dashed line at the 0.001 target
plt.xlabel("Epoch")
plt.ylabel("Training error (mean of (d - y)$^2$)")
plt.title("Learning curves")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "Task2_learning_curves.png"), dpi=150, bbox_inches="tight")

plt.show()                                # open both figure windows; script waits until they're closed

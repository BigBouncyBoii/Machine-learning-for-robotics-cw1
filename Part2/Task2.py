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

# %% Settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

ETA = 0.1            # SGD learning rate
N_EPOCHS = 3000
SEED = 67            # reset at the start of every training run
ERR_TARGET = 0.001   # report the first epoch below this

ALPHA = 0.005        # ADAM settings
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8

FORGET_DERIVATIVE = False   # what-if: replace the tanh slope 1 - a^2 by 1
OUT_DIR = "results_forgotten_derivative" if FORGET_DERIVATIVE else "results_normal"
os.makedirs(OUT_DIR, exist_ok=True)

# %% Load data
df = pd.read_csv("RegressionData.csv")
x, d = df["x"].values, df["d"].values          # (41,), (41,)

xtest = np.linspace(-0.97, 0.93, 20)


def reset_weights():
  """Set W1, b1, W2, b2 back to the given starting weights."""
  global W1, b1, W2, b2
  w = np.load("InitialWeights.npz")
  W1, b1 = w["W1r"], w["b1r"]                  # (3,1), (3,1)
  W2, b2 = w["W2r"], w["b2r"]                  # (1,3), (1,1)


reset_weights()

# %% Forward pass
def forward(xin):
  """xin: scalar or 1-D array of N inputs. Returns a1 (3,N) and y (1,N)."""
  xin = np.reshape(xin, (1, -1))               # (1,N)
  a1 = np.tanh(W1 @ xin + b1)                  # hidden outputs, (3,N)
  y = W2 @ a1 + b2                             # linear output, (1,N)
  return a1, y


def training_error():
  """Mean of (d - y)^2 over all 41 training samples."""
  _, y = forward(x)
  return np.mean((d - y.ravel()) ** 2)


# %% Deltas
def deltas(di, a1, y):
  """Return delta1 (3,1) for the hidden nodes and delta2 (1,1) for the output node."""
  # output delta: delta2 = ∂E/∂n2 = ∂E/∂y · ∂y/∂n2, with E = ½(d - y)²
  dEdy = -(di - y)
  dydn2 = 1                        # linear output, y = n2
  delta2 = dEdy * dydn2            # (1,1)

  # hidden delta: delta1_j = ∂E/∂n1_j = delta2 · ∂n2/∂a1_j · ∂a1_j/∂n1_j
  dn2da1 = W2.T                    # (3,1), one weight W2_j per hidden node
  da1dn1 = 1 if FORGET_DERIVATIVE else 1 - a1**2   # (3,1), slope of tanh
  delta1 = delta2 * dn2da1 * da1dn1   # (3,1)
  return delta1, delta2


def gradients(xi, di, a1, y):
  """Return dE/dW1, dE/db1, dE/dW2, dE/db2 for one sample (same shapes as the weights)."""
  delta1, delta2 = deltas(di, a1, y)
  # gradient = delta of the node the weight feeds into x input to that weight
  dn2dw2 = a1.T                  # (1,3)
  dEdw2 = delta2 * dn2dw2        # (1,3)
  dEdb2 = delta2                 # (1,1), input to a bias is 1

  dn1dw1 = xi                    # scalar, the network input
  dEdw1 = delta1 * dn1dw1        # (3,1)
  dEdb1 = delta1                 # (3,1)
  return dEdw1, dEdb1, dEdw2, dEdb2


# %% Update
def update_sgd(xi, di, a1, y):
  """One SGD step: w <- w - eta * dE/dw for every weight and bias."""
  global W1, b1, W2, b2
  dEdw1, dEdb1, dEdw2, dEdb2 = gradients(xi, di, a1, y)
  W2 = W2 - ETA * dEdw2
  b2 = b2 - ETA * dEdb2
  W1 = W1 - ETA * dEdw1
  b1 = b1 - ETA * dEdb1


def train_sgd():
  """Train from the given weights with SGD. Returns the training error after each epoch."""
  reset_weights()
  np.random.seed(SEED)
  err = np.zeros(N_EPOCHS)
  for epoch in range(N_EPOCHS):
    for i in np.random.permutation(len(x)):      # shuffle every epoch: indices 0..40 in a random order
      a1, y = forward(x[i])
      update_sgd(x[i], d[i], a1, y)              # update after every sample
    err[epoch] = training_error()
  return err


# %% ADAM
def update_adam(xi, di, a1, y, m, v, t):
  """One ADAM step. m and v are lists of 4 arrays (one per weight/bias), updated in place; t is the step count."""
  global W1, b1, W2, b2
  grads = gradients(xi, di, a1, y)
  params = [W1, b1, W2, b2]

  for k in range(4):
    g = grads[k]
    m[k] = BETA1 * m[k] + (1 - BETA1) * g          # 1st moment: running mean of gradients
    v[k] = BETA2 * v[k] + (1 - BETA2) * g**2       # 2nd moment: running mean of squared gradients
    m_hat = m[k] / (1 - BETA1**t)                  # bias correction (m and v start at zero)
    v_hat = v[k] / (1 - BETA2**t)
    params[k] = params[k] - ALPHA * m_hat / (np.sqrt(v_hat) + EPS)
  W1, b1, W2, b2 = params


def train_adam():
  """Train from the given weights with ADAM. Returns the training error after each epoch."""
  reset_weights()
  np.random.seed(SEED)
  m = [np.zeros_like(p) for p in (W1, b1, W2, b2)]   # m, v and t start at zero
  v = [np.zeros_like(p) for p in (W1, b1, W2, b2)]
  t = 0
  err = np.zeros(N_EPOCHS)
  for epoch in range(N_EPOCHS):
    for i in np.random.permutation(len(x)):      # shuffle every epoch
      a1, y = forward(x[i])
      t += 1                                     # one step per sample, never reset between epochs
      update_adam(x[i], d[i], a1, y, m, v, t)
    err[epoch] = training_error()
  return err


# %% Run
err_sgd = train_sgd()
_, y_test_sgd = forward(xtest)                   # SGD fit, saved before ADAM changes the weights

err_adam = train_adam()
_, y_test_adam = forward(xtest)

# %% Plots
def first_below(errors, target):
  below = np.nonzero(errors < target)[0]
  return f"epoch {below[0] + 1}" if below.size else "not reached"


print(f"SGD  final training error: {err_sgd[-1]:.6f}")
print(f"ADAM final training error: {err_adam[-1]:.6f}")
print(f"SGD  first below {ERR_TARGET}: {first_below(err_sgd, ERR_TARGET)}")
print(f"ADAM first below {ERR_TARGET}: {first_below(err_adam, ERR_TARGET)}")

plt.figure()
plt.plot(x, d, "o", label="Training data")
plt.plot(xtest, y_test_sgd.ravel(), "x-", label="SGD output on xtest")
plt.plot(xtest, y_test_adam.ravel(), "+--", label="ADAM output on xtest")
plt.xlabel("x")
plt.ylabel("d, y")
plt.title("1-3-1 network fit")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "Task2_fit.png"), dpi=150, bbox_inches="tight")

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

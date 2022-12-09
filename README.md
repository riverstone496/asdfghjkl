# ASDL: Automatic Second-order Differentiation Library

NOTE: this branch `dev-grad-maker` is under development and will be merged to `master` branch soon.

ASDL is an extension library of PyTorch to easily 
calculate **Second-Order Matrices**
and apply **Gradient Preconditioning** for deep neural networks.

## Gradient Preconditioning in Deep Learning

![image](https://user-images.githubusercontent.com/7961228/206647457-7a46d101-941f-4de4-9610-10b72011f01c.png)

```diff
import torch
from asdl.precondition import PreconditioningConfig, KfacGradientMaker

# Initialize model
model = Net()

# Initialize optimizer (SGD is recommended)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Initialize KfacGradientMaker
config = PreconditioningConfig(data_size=batch_size, damping=0.01)
gm = KfacGradientMaker(model, config)

# Training loop
for x, t in data_loader:
-    y = model(x)
-    loss = loss_fn(y, t)
-    loss.backward()

+    dummy_y = gm.setup_model_call(model, x)
+    gm.setup_loss_call(loss_fn, dummy_y, t)
+    y, loss = gm.forward_and_backward()

```

## Installation

```shell
git clone https://github.com/kazukiosawa/asdl.git
cd asdl
pip install -e .
```
(`asdl` will be available in PyPI soon.)

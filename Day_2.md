# Day 2
## Weight Space Linearization [Colab NB 2]:(https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/weight_space_linearization.ipynb#scrollTo=-W1ws1B-6_vq)
### Summary
* Comparison between regular loss and linear network loss. 
* To linearize the network, we use: `f_lin = nt.linearize(f, params)` where f is the apply_fn of model and parama are from prior
* Loss function is `loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)` and gradient linear loss fn is `grad_lin_loss = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))`
* The first lin_state is the saame is state but is different aftwards for every training step as f_lin func is involved.
* There is not much rough difference in loss and linear_loss
## Space Function Linearization [COLAB NB 3]:(https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/function_space_linearization.ipynb#scrollTo=8KPv0bOW6UCi)
### Summary
* Constructing the NTK
    ```
    ntk = nt.batch(nt.empirical_ntk_fn(f), batch_size=16, device_count=0)
    
    g_dd = ntk(train['image'], None, params)
    g_td = ntk(test['image'], train['image'], params)
    ```
* Comparisons btw
  1. Gradient descent, MSE loss
    ```
    predictor = nt.predict.gradient_descent_mse(g_dd, train['label'], learning_rate=learning_rate)
    ```
  2. Gradient descent, Cross entropy loss
    ```
    predictor = nt.predict.gradient_descent(loss, g_dd, train['label'], learning_rate=learning_rate)
    ```
  3. Momentum, Cross entropy loss
    ```
    mass = 0.9
    opt_init, opt_apply, get_params = momentum(learning_rate, mass) 
    state = opt_init(params)
    ```
    ```
    predictor = nt.predict.gradient_descent(loss, g_dd, train['label'], learning_rate=learning_rate, momentum=mass)
    ```
## Phase Diagram [COLAB NB 4]:(https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/phase_diagram.ipynb)
### Summary
* 








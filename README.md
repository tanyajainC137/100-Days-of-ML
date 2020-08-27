# 100-Days-of-ML
The 100 days of Machine Learning Challenge
## Pre-requisites
* Knowledge of Statistics, Calculus and Probability (matrix algebra ofc) 
* Intro to ML, Python and Python libraries like numpy, pandas, scikit-learn
* ML Algos like Regression, Decision Trees, SVM, KNN, K-means clustering and Association 
* Textual manipulation like bag of words, word-counter, encoding, word-vectors, lemmitization, stoppwords 
* Understanding of Deep learning concepts like perceptron, feed forward neural network, layers, back-propagation, random weights initialisation, non-linear activations, optimization algos like gradient descent, learning rate  
* **Most importantly a keen spirit to participate in the evolving of the AI and always striding towards building more capable AI**
## Day 0, 24th Aug 2020
Did some prelim research on upcoming AI tech and their methods and approaches
* Gaussian Processes in Machine Learning; neural tangents
* Multi-modal models; datasets on baidu and waymo
* Multi-tasking models; by means of Graph based networks
* Creative AI; by GANs
* Focus on the operational aspects of ML ; MLflow
* Quantization of NN; Graffitist
* Quantum computing and NNs; computational supremacy
* Ethical aspects of DL; XAI  
> Will delve into more depth of each domain further

## Day 1, 25th Aug 2020
### Gaussian Processes in Machine Learning: 
1. YT lectures on GP : https://www.youtube.com/watch?v=4vGiHC35j9s
2. google/neural-tangents framework combines neural networks with GPs and come up with predictions that have an uncertanity factor attached with them.
3. Libraries: 
  - JAX
  - neural_tangents  
4. Forked the GH repo of neural-tangents and started practicing on provided colab notebooks. 
 
### Key takeaways from [Colab NB #1]:(https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb#scrollTo=c3lXqB1t3U9g):
  
  1. specialised random number generator;  
      * `key = jax.random.PRNGKey(seed);` 
      * `key, key1, key2 = jax.split(key, 3)`
  
  2. NN easy comparison by:
      * Regular NN from jax.experimental.stax this model returns two functions:  
          ` init_fn(key, input_shape)`
          ` apply_fn(params, xs)`  
      * neural_tangents.stax for NN with GP, this model returns three functions:  
          ` init_fn(key, input_shape)`
          ` apply_fn(params, xs)`
          ` kernel_fn(test_xs, test_xs, 'nngp' or 'ntk')`
      
      > kernel_fn used to store the covariance matrices of data;
  
  3. predict_fn is used to make predictions on the test data using gradient descent 
      * `predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_xs, train_ys, diag_reg = 'constt')`
  
  4. further mean and covariance is returned by the predict_fn on selecting the data and method (nngp or ntk)
  ```
      * nngp_mean, nngp_cov = predict_fn(text_xs, get = 'nngp', compute_cov = True)
      * ntk_mean, ntk_cov = predict_fn(test_xs, get = 'ntk' , compute_cov = True)
  ```    
      > the mean is refined by reshaping and cov matrix diagnal on sqrt gives the std_dev
  
  5. the below plot shows the uncertainities associated with the mean value predictions for test_data;
      * `plt.fill_between(test_xs, mean+2*cov, mean-2*cov, alpha = 0.2, color='red')`
  
  6. two methods of creating kernel_fn and predictions: 
      1. 'nngp' (bayesian infinite-width)
      2. 'ntk' (gradient descent)
  
  7. loss computation for finite-time interference
      * `ts = np.linspace(0, 10 **  3, 10 *  -1);  #large set of values `
      * `mean_loss = loss_fn(predict_fn, test_ys or train_ys, ts, test_xs)`
      > default value of ts is infinity and test_xs is used only along test_ys for test_loss_mean
      
  8. training the network
      * comparison of GP-interference and ensemble of finite-widths
      * start with single network drawn from prior and then generalise to ensemble
      
      `learning_rate = 0.1`
      `training_step = 1e4`
      
      `opt_init, opt_update, get_params = optimizers.sgd(learning_rate)`
      ```
      loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y) ** 2))
      grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))
      
      ```
  9. training the ensemble of neural networks
      * using JAX function VMAP to train the ensemble neural network
      * train_network is defined as a function to get params, train_losses and test_losses from ensemble_key
      * ensemble_key is an array of keys of length equal to ensemble_size and is split from the previous key
      ```
      def train_network(key):
        train_losses = []
        test_losses = []

        _, params = init_fn(key, (-1, 1)) 
        opt_state = opt_init(params)

        for i in range(training_steps):
          train_losses += [np.reshape(loss(get_params(opt_state), *train), (1,))]  
          test_losses += [np.reshape(loss(get_params(opt_state), *test), (1,))]
          opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

        train_losses = np.concatenate(train_losses)
        test_losses = np.concatenate(test_losses)
        return get_params(opt_state), train_losses, test_losses
      ```
      ```
      ensemble_size = 100
      ensemble_key = random.split(key, ensemble_size)
      params, train_loss, test_loss = vmap(train_network)(ensemble_key) 
      ```
      * plotting mean of train_loss and mean of test_loss along with ntk_train_loss_mean and ntk_test_loss_mean against ts to get perspective of errors over time
      ```
      mean_train_loss = np.mean(train_loss, axis=0)
      mean_test_loss = np.mean(test_loss, axis=0)
      ```
      * plotting mean and std of ensemble along with ntk_test_mean and ntk_test_std against the test_xs
      ```
      ensemble_fx = vmap(apply_fn, (0, None))(params, test_xs)
      mean_fx = np.reshape(np.mean(ensemble_fx, axis=0), (-1,))
      std_fx = np.reshape(np.std(ensemble_fx, axis=0), (-1,))
      ```
  10. This ensemble neural network inference is valid for any kind of complex network architecture
      * For example:
      ```
      ResBlock = stax.serial(
      stax.FanOut(2),
      stax.parallel(
          stax.serial(
             stax.Erf(),
              stax.Dense(512, W_std=1.1, b_std=0),
              ),
          stax.Identity()
          ),
      stax.FanInSum()
      )  
          
      init_fn, apply_fn, kernel_fn = stax.serial(
      stax.Dense(512, W_std=1, b_std=0),
      ResBlock, ResBlock, stax.Erf(),
      stax.Dense(1, W_std=1.5, b_std=0))
      ```
  

    
  


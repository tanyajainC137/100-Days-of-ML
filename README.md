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
* Gaussian Processes in Machine Learning: 
* Multi-modal models; datasets on baidu and waymo
* Multi-tasking models; by means of Graph based networks
* Creative AI; by GANs
* Focus on the operational aspects of ML ; MLflow
* Quantization of NN; Graffitist
* Ethical aspects of DL; XAI
Will delve into more depth of each domain further

## Day 1, 25th Aug 2020
### Gaussian Processes in Machine Learning: 
1. YT lectures on GP : https://www.youtube.com/watch?v=4vGiHC35j9s
2. google/neural-tangents framework combines neural networks with GPs and come up with predictions that have an uncertanity factor attached with them.
3. Forked the GH repo of neural-tangents and practicing on provided colab notebooks. 
  Main aspects: specialised random number generator; kernel_fn used to store the covariance matrices of data; plt.fill_between(mean_fn, 2*cov, -2*cov, alpha = 0.2, color='red')     this plot shows the uncertainities associated with the mean value predictions for test_data; two methods of creating kernel_fn: 'nngp' (bayesian infinite-width) , 'ntk' (gradient descent).
  
  

    
  


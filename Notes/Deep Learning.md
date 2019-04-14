# Deep Learning
## Intro to Neual Network
### Perceptron
![](/img/DL/1.png)

#### Algorithm:
For a point with coordinates (p,q)(p,q), label yy, and prediction given by the equation $\hat{y} = step(w_1 x_1 + w_2 x_2 + b)$:
* If the point is correctly classified, do nothing.
* If the point is **classified positive**, but it has a **negative label**, **subtract** $\alpha p, \alpha q$ and $\alpha\ to\ w_1, w_2\ and\ b$ respectively.
* If the point is **classified negative**, but it has a **positive label**, **add** $\alpha p, \alpha q$ and $\alpha\ to\ w_1, w_2\ and\ b$ respectively.

### Discrete vs. Continuous
![](/img/DL/2.png)
Sigmoid function: returns the probability of the result.
$$\sigma(x) = \frac{1}{1 + e^{-x}}, $$
where x is the predict value of the model ($y_{pred}$).

### Softmax Function
$$Softmax = \frac{e^{Zi}}{e^{Z1} + e^{Z2} + ... + e^{Zn}}$$

When n = 2, softmax = sigmid.

### One-Hot Encoding
![](/img/DL/3.png)

### Maximum Likelihood
* Find a model which classifies most points correctly with P(all) indicating how accurate the model is.
* Better: use Cross-Entropy

## Cross-Entropy
* $$CE = -\sum_{i = 1}^{n} ln(P_i)$$
* Objective: Minimize CE.

## Binary-Class Cross Entropy
* $$CE = -\sum_{i = 1}^{n} y_i * ln(P_i), y_i = \{0, 1\}$$

## Multi-Class Cross Entropy
* $$CE = -\sum_{i=1}^{n} \sum_{j=1}^{m} y_{ij} ln(p_{ij})$$
![](/img/DL/4.png)

## Logistic Regression
### Binary Classification
* $$Error = - \frac{1}{m} \sum_{i=1}^{m} (1 - y_i) * ln(1 - \hat{y_i}) + y_i * ln(\hat{y_i})$$
* $$Since\ \hat{y_i} = \sigma(Wx^{(i)} + b),$$
* $$E(W, b) = - \frac{1}{m} \sum_{i=1}^{m} (1 - y_i) * ln(1 - \sigma(Wx^{(i)} + b)) + y_i * ln(\sigma(Wx^{(i)} + b))$$

### Multi-class Classification
* $$Error = - \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{n} y_{ij} * ln(\hat{y_{ij}})$$

## Gradient Descent
For Binary Classification:
* $$Error = - \frac{1}{m} \sum_{i=1}^{m} (y_i ln(\hat{y_i}) + (1 - y_i) ln(1 - \hat{y_i}))$$

* $$\nabla{E} = (\frac{\partial}{\partial{w_1}} E, \cdots, \frac{\partial}{\partial{w_n}} E, \frac{\partial}{\partial{b}} E)$$

* $$\frac{\partial}{\partial{w_j}} \hat{y} = \hat{y} (1 - \hat{y}) x_j$$

* $$\frac{\partial}{\partial{w_j}} E = \frac{\partial}{\partial{w_j}} [-y log(\hat{y}) - (1 - y) log(1 - \hat{y})] = -(y - \hat{y}) x_j$$

* $$\frac{\partial} {\partial{b}} E = - (y - \hat{y})$$

* $$\nabla{E} = - (y - \hat{y})(x_1, \cdots, x_n, 1)$$

## Logistic Regression Algorithm
![](/img/DL/5.png)

## Non-Linear Models
![](/img/DL/6.png)

### Neural Network Architecture
![](/img/DL/7.png)
![](/img/DL/8.png)
![](/img/DL/9.png)
![](/img/DL/10.png)

### Multiple Layers
![](/img/DL/11.png)
![](/img/DL/12.png)

### Feedforward
* The process neural networks use to turn the input into an output. Let's study it more carefully, before we dive into how to train the networks.

### Backpropagation
* Doing a feedforward operation.
* Comparing the output of the model with the desired output.
* Calculating the error.
* Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
* Use this to update the weights, and get a better model.
* Continue this until we have a model that is good.
![](/img/DL/13.png)

## Gradient Descent: The math
* $$E = \sum_{\mu} (y^{\mu} - \hat{y}^{\mu})^2$$
* $$\hat{y}^{\mu} = f(\sum_{i} w_i x_i^{\mu})$$
* $$E = \sum_{\mu} (y^{\mu} - f(\sum_{i} w_i x_i^{\mu}))^2$$
* $$\Delta w = -gradient$$
* $$\Delta w \propto -\frac{\partial E}{\partial w_i}$$
* $$\Delta w = - \eta \frac{\partial E}{\partial w_i}, \eta\ is\ Learning\ Rate$$
* $$\frac{\partial E}{\partial w_i} = \frac{\partial}{\partial w_i} \frac{1}{2} (y - \hat{y})^2 = (y - \hat{y}) \frac{\partial}{\partial {w_i}} (y - \hat{y}) = -(y - \hat{y}) \frac{\partial{\hat{y}}}{\partial{w_i}} = -(y - \hat{y})\frac{\partial}{\partial{w_i}}\sum_{i}w_i x_i$$
* $$\frac{\partial}{\partial{w_i}}\sum_{i}w_i x_i = x_i$$
* $$\frac{\partial E}{\partial w_i} = - (y - \hat{y}) f'(h) x_i$$
* $$\Delta w_i = \eta (y - \hat{y}) f'(h) x_i$$
* Define "Error Term" **$\delta$**:
$$\delta = (y - \hat{y}) f'(h)$$
(f(h) is active function(such as sigmoid)
$$w_i = w_i + \eta \delta x_i$$
![](/img/DL/14.png)
![](/img/DL/15.png)

### Initialize weights
* normal distribution centered at 0, good value: $\frac{1}{\sqrt{x}}$

## Regularization
### L1 Regularization (LASSO)
$$Error Function = -\frac{1}{m} \sum_{i=1}^{m} (1 - y_i) ln(1 - \hat{y_i}) + y_i ln(\hat{y_i}) + \lambda(\|w_1\| + \cdots + \|w_n\|)$$
Good for Feature Selection

### L2 Regularization (Ridge)
$$Error Function = -\frac{1}{m} \sum_{i=1}^{m} (1 - y_i) ln(1 - \hat{y_i}) + y_i ln(\hat{y_i}) + \lambda(w_1^2 + \cdots + w_n^2)$$
Normally better for training Models

## Local Minima
![](/img/DL/16.png)
Solution: Random Restart

## Vanishing Gradient
Solution:
Change activation function
* tanh(x)

![](/img/DL/17.png)

* relu(x)

![](/img/DL/18.png)

## Batch vs Stochastic Gradient Descent
### Batch Gradient Descent
In each epoch we take all of our training data and run through the entire neural network, find prediction, calculate error, and back-propagate.
### Stochastic Gradient Descent
Split the data into several batches, calculate the gradient of the error function based on those points and then move one step in that direction.

## Learning Rate
Rule:
If steep: long steps
If plain: small steps

## Momentum $\beta$
![](/img/DL/19.png)

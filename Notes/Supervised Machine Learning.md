# Supervised Machine Learning
## Supervised Machine Learning
* Algorithms learn from labeled data
* Two Categories: **Classification** and **Regression**

## Linear Regression
### Absolute Trick
* Given a point p and a line y = w_1 * x + w_2

* How to change the equation:
$y' = (w_1 + p * \alpha) * x + (w_2 + 1 * \alpha), \alpha$ is learning rate.
**If the point is below the line, the intercept decreases, else, increases. If the point has a negative x-value, the slope increases, else decreases.**

![](/img/1.png)

![](/img/2.png)

### Square Trick
* Given a point p and a line $y = w_1 * x + w_2$, p' is a point on the line which $p_x = {p'}_x.$
* How to change the equation:
$y' = (w_1 + p(q - q') * \alpha) * x + (w_2 + (q - q') * \alpha), \alpha$ is learning rate. **No rules**.

![](/img/3.png)

### Mean Absolute Error
* Definition
$$\hat{w}_{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y - \hat{y}|$$
![](/img/4.png)

### Mean Squared Error
* Definition
$$\hat{w}_{MSE} = \frac{1}{2m} \sum_{i=1}^{m} (y - \hat{y})^2$$
![](/img/5.png)

### Minimizing Error Functions
For mean squred error,
$$\hat{w}_{MSE} = \frac{1}{2m} \sum_{i=1}^{m} (y - \hat{y})^2$$

Also, we defined the prediction to be
$$\hat{y} = w_1 x + w_2$$

So to calculate the derivative of the Error with respect to $w_1$,
$$\frac{\partial Error}{\partial w_1} = \frac{\partial Error}{\partial \hat{y}} \frac{\partial \hat{y}} {\partial w_1} = -(y - \hat{y}) x$$

the derivative of the Error with respect to $w_2$,
$$\frac{\partial Error}{\partial w_2} = \frac{\partial Error}{\partial \hat{y}} \frac{\partial \hat{y}} {\partial w_2} = -(y - \hat{y})$$

### Gradient Descent
* Avoid calculating n equations with n variables using Linear Algebra, which is time consuming.

* iterate $w_i$ :
$$w_i \rightarrow w_i - \alpha \frac{\partial} {\partial w_i} Error$$

### Mini-Batch Gradient Descent
* Split your data into many small batches. Each batch, with roughly the same number of points. Then, use each batch to update your weights.

### Linear Regression Warnings
* Linear Regression Works Best When the Data is Linear
* Adding a few points that are outliers and don't fit the pattern really changes the way the model predicts.

### Polynomial Regression
* $$\hat{y} = w_1 x^3 + w_2 x^2 + w_1 x + w_0$$

### Regularization
#### 1. L1 Regularization (LASSO)
$$\hat{w}_{ridge} = arg min_w \sum_{i=1}^{n}(y_i - x_i^Tw)^2 + \lambda \|w\|_1$$
where $\lambda$ is hyperparameter and $\|w\|_1$ is L1-norm of the $w$ vector.

#### 2. L2 Regularization (Ridge)
$$\hat{w}_{ridge} = arg min_w \sum_{i=1}^{n}(y_i - x_i^Tw)^2 + \lambda \|w\|^2_2$$
where $\lambda$ is hyperparameter and $\|w\|^2_2$ is L2-norm of the $w$ vector.

#### LASSO & Ridge comparison
| L1 Regularization | L2 Regularization     |
| :------------- | :------------- |
| Computationally Inefficient (unless data is sparse)| Computationally efficient|
|Sparse Outputs| Non-Sparse Outputs|
|Feature Selection| No Feature Selection|

### Feature Scaling
#### Standardizing (the most common)
* Standardizing is completed by taking each value of your column, subtracting the mean of the column, and then dividing by the standard deviation of the column.
* Example:

$$df['height\_standard'] = \frac{df['height'] - mean(df['height'])}{std(df['height'])}$$
#### Normalizing
* With normalizing, data are scaled between 0 and 1.

$$df['height\_normal'] = \frac{df['height'] - min(df['height'])}{max(df['height']) - min(df['height'])}$$

#### When to use
1. When your algorithm uses a distance-based metric to predict. (Support Vector Machines(SVM) / K-Nearest Neighbors(KNN))
2. When you incorporate regularization. (LASSO / Ridge)

## Perceptron
### 2D:
![](\img\6.png)
Boundary Line:

$w_1x_1 + w_2x_2 + b = 0$

$\Rightarrow Wx + b = 0, W = (w_1, w_2), x = (x_1, x_2)$
y = 0 or 1 (label)

Prediction:
$$\hat{y} = \left\{\begin{array}{cc}
       		1, & \text{if}\ Wx + b \geqslant 0\\
       		0, & \text{if}\ Wx + b < 0
\end{array}\right.$$

### Higher Dimensions:
#### 1. 3D
Line $\rightarrow$ Plane

$w_1x_1 + w_2x_2 + w_3x_3 + b = 0$  ($Wx + b = 0$)

Prediction:
$$\hat{y} = \left\{\begin{array}{cc}
       		1, & \text{if}\ Wx + b \geqslant 0\\
       		0, & \text{if}\ Wx + b < 0
\end{array}\right.$$

#### 2. nD
Line $\rightarrow$ n-1 dimensional hyperplane
$w_1x_1 + w_2x_2 + w_3x_3 + ...+ w_nx_n + b = 0$  ($Wx + b = 0$)

Prediction:
$$\hat{y} = \left\{\begin{array}{cc}
       		1, & \text{if}\ Wx + b \geqslant 0\\
       		0, & \text{if}\ Wx + b < 0
\end{array}\right.$$

### Perceptron
![](\img\7.png)
![](\img\8.png)

* AND Perceptron
![](\img\9.png)

* OR Perceptron
![](\img\10.png)

* Perceptron Algorithm
![](\img\11.png)

## Decision Tree
### Entropy
![](\img\12.png)

#### Two-class Entropy
General Formula for entropy when the balls can be of two colors.
![](\img\13.png)
$$Entropy = -\frac{m}{m + n} log_2(\frac{m}{m + n}) - \frac{n}{m + n} log_2(\frac{n}{m + n})$$

#### Multi-class Entropy
$$Entropy = -p_1 log_2(p_1) - p_2 log_2(p_2) - ... - p_n log_2(p_n) = \sum_{i=1}^{n} p_i\ log_2(p_i)$$

#### Information Gain Formula
$$Information\ Gain = Entropy(Parent) - [p1 * Entropy(Child_1) + p2 * Entropy(Child_2)]$$

#### Hyperparameters for Decision Trees
##### 1. **Maximum Depth**
The maximum depth of a decision tree is simply the largest possible length between the root to a leaf. A tree of maximum length $k$ can have at most $2^k$ leaves.
![](\img\14.png)
##### 2. **Minimum number of samples to split**
A node must have at least `min_samples_split` samples in order to be large enough to split. If a node has fewer samples than `min_samples_split` samples, it will not be split, and the splitting process stops.

![](\img\15.png)
##### 2. **Minimum number of samples per leaf**
When splitting a node, one could run into the problem of having 99 samples in one of them, and 1 on the other. This will not take us too far in our process, and would be a waste of resources and time. If we want to avoid this, we can set a minimum for the number of samples we allow on each leaf.
![](\img\16.png)

## Naive Bayes
### Bayes Theorem
| Known | Inferred     |
| :------------- | :------------- |
| P(A) <br/> P(R\|A)| P(A\|R)|

![](\img\17.png)

### Naive Bayes Algorithm
$P(A \bigcap B) = P(A) * P(B)$

$P(A | B) P(B) = P(B | A) P(A)$

$P(A|B) \propto P(B | A) P(A)$

**Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).

$\frac{number\ of\ correct\ predictions}{total\ number\ of\ predictions}$

**Precision** tells us what proportion of messages we classified as spam, actually were spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classification), in other words it is the ratio of

`[True Positives / (True Positives + False Positives)]`

**Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of

`[True Positives / (True Positives + False Negatives)]`

## Support Vector Machines (SVM)
### Minimizing Distances
![](\img\18.png)

### Error Function Intuition
![](\img\19.png)
![](\img\20.png)

### Perceptron Algorithm
![](\img\21.png)

![](\img\22.png)

### Classification Error
![](\img\23.png)

### Margin Error
![](\img\24.png)

### Margin Error Calculation
$W = (w_1, w_2), x = (x_1, x_2), Wx = w_1x_1 + w_2x_2$

We have three lines:
* $Wx + b = 1$
* $Wx + b = 0$
* $Wx + b = -1$

To calculate the distance between the first and the third, we only need to find the distance between the first two, and multiply by two.

We move these lines so that $Wx + b = 0$ is translated to $Wx = 0$

![](img/25.png)

Find the intersect: assume the intersect is $(p,q)$
$$\hat{y} = \left\{\begin{array}{cc}
       		w_1p + w_2q = 1, \\
       		(p, q) = k(w_1, w_2),
\end{array}\right.$$

$\Rightarrow k({w_1}^2 + {w_2}^2) = 1$
$\Rightarrow k = \frac{1}{{w_1}^2 + {w_2}^2} = \frac{1}{|W|^2}$

Therefore, the blue point represents the vector $\frac{(w_1, w_2)}{|W|^2}$

![](img/26.png)

Therefore the total distance is $\frac{2}{|W|}$

### C parameter
Error = C * Classification Error + Margin Error
![](img/27.png)

### Polynomial Kernel
#### Kernel Trick
How to use SVM to separate these points?
![](img/28.png)

Adding a y axis
![](img/29.png)

![](img/30.png)

![](img/31.png)

![](img/32.png)

### RBF Kernel
![](img/33.png)

Find a line to separate red and blue points

![](img/34.png)

add radial basis functions for each point
![](img/35.png)

![](img/36.png)

![](img/37.png)

How tall is the mountain at that point?

![](img/38.png)

![](img/39.png)

![](img/40.png)

![](img/41.png)

Hyperparameter $\gamma$
2D:
![](img/42.png)
3D:
![](img/43.png)

![](img/44.png)

What is $\gamma$?

Normal Distribution: $y = \frac{1}{\sigma \sqrt{2 \pi}} e^{-\frac{(x - \mu) ^ 2}{2 \sigma^2}}$
$\gamma = \frac{1}{2 \sigma ^ 2}$

![](img/45.png)

## Ensemble Methods
### Ensemble
Why?

Two competing variables in finding a well fitting machine learning model: Bias and Variance

* Bias: an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

* Variance: an error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting).

The model will perform better by meeting in the middle in terms of bias and variance.

Two ways:

* **Bootstrap the data** - sampling the data with replacement and fitting your algorithm to the sampled data.

* **Subset the features** - in each split of a decision tree or with each algorithm used an ensemble only a subset of the total possible features are used.

### Random Forests
Train Data:

![](img/46.png)

Pick subsets from Train Data:

$1^{st} subset:$
![](img/47.png)

$2^{nd} subset:$
![](img/48.png)

$3^{rd} subset:$
![](img/49.png)

Combine results:
![](img/50.png)
![](img/51.png)

### AdaBoost
![](img/52.png)

Find a line to maximize accuracy, and punish the misclassified points more

![](img/53.png)

The next step is trying to clssify these points right, so that the accuracy will be the maximum.

![](img/54.png)

![](img/55.png)

Voting:

![](img/56.png)

![](img/57.png)

![](img/58.png)

### Weighing the data(AdaBoost)
![](img/59.png)

First model

![](img/60.png)

![](img/61.png)

![](img/62.png)

### Weighing the models(AdaBoost)
![](img/63.png)

![](img/64.png)

$weight = ln (\frac{accuracy}{1 - accuracy})$

### Combining the models
![](img/65.png)

![](img/66.png)

![](img/67.png)

![](img/68.png)

![](img/69.png)

![](img/70.png)

## Model Evaluation Metrics
### Confusion Matrix
* **Type 1 Error** (Error of the first kind, or False Positive): In the medical example, this is when we misdiagnose a healthy patient as sick.

* **Type 2 Error** (Error of the second kind, or False Negative): In the medical example, this is when we misdiagnose a sick patient as healthy.

* Accuracy:

$Accuracy = \frac{True Positive + False Negative}{overall}$

* Recall
False Negative is important
$Recall =  \frac{True Positive}{True Positive + True Negative}$

* Precision
False positive is important
$Precision =  \frac{True Positive}{False Positive + True Positive}$

### F1 Score(Harmonic Mean)

$F1 Score = 2 \frac{Precision * Recall}{Precision + Recall}$

### $F_{\beta}\ Score$
$F1 Score = (1 + \beta^2) \frac{Precision * Recall}{\beta^2 * Precision + Recall}$

if $\beta = 0,$ we get Precision

if $\beta = \infty,$ we get Recall

### Roc Curve
![](img/71.png)

$True Positive Rate = \frac{True\ Positives}{All\ Positives}$

$False Positive Rate = \frac{False\ Positives}{All\ Negatives}$

![](img/72.png)

![](img/73.png)

![](img/74.png)

### Regression Metrics

R2 Score = $1 - \frac{error_{linear\ regression}}{error_{simple}}$

![](img/75.png)

## Training and Tuning
### Type of Errors
#### Underfitting
* **Does not do well in the training set**
* **Error due to bias**

![](img/76.png)

#### Overfitting
* **Does well in the training set**, but it tends to memorize it instead of learning the characteristics of it.
* **Error due to variance**

![](img/77.png)

#### Summary

![](img/78.png)

### Cross Validation
#### Model Complexity Graph
![](img/79.png)
**Important: This is not allowed because we use test data to train our model.**

Introduce Cross Validation
![](img/80.png)

![](img/81.png)

![](img/82.png)

### K-fold Cross Validation
![](img/83.png)

### Learning Curves
![](img/84.png)

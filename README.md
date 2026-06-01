# Linear Regression Project - Car Price Predictor

This project implements a **univariate linear regression** model (with a single predictive feature) to estimate the price of a used car based on its mileage. Parameter optimization is achieved using the **batch gradient descent** algorithm.

The project is split into two distinct programs: a training script that computes and saves the optimal parameters, and a prediction script that utilizes these parameters to output a price estimation to the user.

---

## Mathematical Formulation & Algorithm

### Hypothesis Function
The price estimation is modeled by the following affine function:

$$\text{estimatePrice}(\text{mileage}) = \theta_0 + (\theta_1 \times \text{mileage})$$

Where:
* $\theta_0$ represents the y-intercept (bias).
* $\theta_1$ represents the slope of the line (slope coefficient).

Before running the training program for the first time, both parameters $\theta_0$ and $\theta_1$ are initialized to $0$.

### Cost Function (Mean Squared Error)
To measure the model's accuracy on a dataset of $m$ samples, we calculate the cost function $E(\theta_0, \theta_1)$:

$$E(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{mileage}[i]) - \text{price}[i])^2$$

### Gradient Descent
To minimize this cost, the algorithm computes the partial derivatives of the cost function with respect to each parameter to determine the gradient vector (the direction of the steepest ascent of the error):

$$\frac{\partial E}{\partial \theta_0} = \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{mileage}[i]) - \text{price}[i])$$

$$\frac{\partial E}{\partial \theta_1} = \frac{1}{m} \sum_{i=0}^{m-1} (\text{estimatePrice}(\text{mileage}[i]) - \text{price}[i]) \times \text{mileage}[i]$$

The parameters are updated **simultaneously** at each iteration according to the following rules:

$$\theta_0 \leftarrow \theta_0 - \alpha \times \frac{\partial E}{\partial \theta_0}$$

$$\theta_1 \leftarrow \theta_1 - \alpha \times \frac{\partial E}{\partial \theta_1}$$

Where $\alpha$ denotes the learning rate. This iterative process repeats until convergence (when the variation in error between two consecutive iterations falls below a predefined threshold $\epsilon$).


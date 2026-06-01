# Linear Regression Project - Car Price Predictor

This project implements a **univariate linear regression** model (with a single predictive feature) to estimate the price of a used car based on its mileage. Parameter optimization is achieved using the **batch gradient descent** algorithm.

The project is split into two distinct programs: a training script that computes and saves the optimal parameters, and a prediction script that utilizes these parameters to output a price estimation to the user.

---

## 1. Mathematical Formulation & Algorithm

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

---

## 2. Project Structure

The project incorporates a critical **Min-Max normalization** step on the mileage data to fix scale disproportions between variables, ensuring numerical stability for the gradient descent during training. The parameters are then geometrically denormalized for real-world application.

The repository contains the following components:
* `train_model.py`: Loads the data from `data.csv`, applies normalization, executes the gradient descent loop, performs the geometric denormalization of the coefficients, and serializes the final $\theta_0$ and $\theta_1$ parameters into a persistent JSON file.
* `predict.py`: An interactive program. It loads the JSON file containing $\theta_0$ and $\theta_1$, prompts the user to input a mileage via the terminal, validates data integrity, and outputs the estimated price rounded to the nearest integer.
* `data.csv`: The dataset file containing the explanatory variable `km` and the target variable `price`.
* `parameters.json`: A file generated after training that stores the serialized values of the theta parameters.

---

## 3. Usage Instructions

### Step 1: Train the Model
Run the training script to process the data, optimize the parameters, and generate the configuration file:
```bash
python3 train_model.py
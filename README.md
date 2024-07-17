# Working with Multiple Input Linear Models to Make Predictions
Overview
This repository contains a case study that demonstrates working with multiple input linear models for predictions. I use randomly generated data and the statsmodels library in Python to fit linear regression models, interpret the results and visualize the relationships between inputs and outputs. It uses two input features to interpret prediction results by generating random data, fitting a linear regression model using the statsmodels interface, and interpreting the results through coefficient summaries and specialized graphics.

## Real-World Applications
- Economics: Predicting economic indicators such as GDP, inflation, and unemployment rates based on multiple inputs like interest rates, consumer spending, and investment levels.
- Financial Forecasting: Predicting stock prices or market trends based on multiple economic indicators. Forecasting stock prices or financial performance of companies based on multiple financial indicators and market variables.
- Healthcare: Estimating patient outcomes based on various health metrics and treatment variables. Predicting patient outcomes based on various input features such as age, blood pressure, cholesterol levels, and other medical data.
- Marketing Analysis: Determining the impact of different marketing strategies on sales performance. Estimating the impact of different marketing strategies on sales by analyzing inputs like advertising spend, pricing strategies, and market trends.
- Environmental Science: Modeling climate change effects using multiple environmental factors. Modeling the impact of various environmental factors on climate change, including temperature, CO2 levels, and other atmospheric conditions.
- Engineering: Predicting system performance based on multiple design and operational parameters.

## Installation
To run the code in this repository, you'll need Python 3 and the following Python packages:
NumPy, Pandas, Matplotlib, Seaborn, Statsmodels

## Usage
To explore the case study, you can follow these steps:
1 Clone the repository:
  git clone https://github.com/RaphRivers/multiple-input-linear-model-case-study.git
  cd multiple-input-linear-model-case-study
2 Open the Jupyter Notebook
3 Run the notebook cells to see the step-by-step process of generating data, fitting the model, and making predictions.

## Methodology
- **Import Modules**
- **Set Regression Coefficients** - Define the coefficients for our linear model.
- **Generate Random Input Data** - Generate random data for our inputs x1 and x2.
- **Calculate Trend and Generate Output** - Using the regression coefficients, calculate the trend and generate the output.
- **Visualize Relationships** - Visualize the relationships between the inputs and output.
- **Fit the Models and Summarize** - Fit the linear model and summarize the results.
- **Visualize Coefficient Summaries** - Visualize the coefficient summaries using a custom function.
- **Making Predictions** - We make predictions using the fitted model.
- **Visualize Predictions** - Visualize the prediction results.


## Importing Modules
The necessary Python modules for this study include NumPy, Pandas, Matplotlib, Seaborn, and Statsmodels.

## Setting Regression Coefficients
The model uses two inputs, x1 and x2, with specified regression coefficients (b0 = -0.25, b1 = 1.95, b2 = 0.2). These coefficients represent the relationship between the predictor variables and the dependent variable.

## Generating Random Input Data
The inputs, x1 and x2, are assumed to have standard normal distributions (mean = 0, standard deviation = 1). A sample size of 35 observations is used, with a reproducible random seed.

## Calculating the Trend
The average output (trend) is calculated using the formula:
$$\mu = \beta_0 + \beta_1 \times x1 + \beta_2 \times x2 $$

## Visualizing Relationships
Scatter plots are used to examine the relationships between inputs (x1 and x2) and output (y). Additionally, regression lines are plotted to identify possible trends.

## Model Fitting and Summarization
Linear models are fitted for each input separately, and their coefficients, standard errors, p-values, and confidence intervals are examined to determine statistical significance.

## Visualizing Coefficient Summaries
A function is used to visualize coefficient summaries with error bars, indicating the significance of each input's effect on the output.

## Linear Additive Features with Multiple Inputs
Both inputs are fitted together in a linear model. The combined effect shows changes in the slope estimates, highlighting the importance of considering all inputs together.

## Making Predictions
Predictions involve creating a data frame with all inputs used to fit the model. The most important input is determined based on the magnitude and scale of the coefficients. Visualization is done using line colors to show the prediction changes across inputs.

## Visualizing Multiple Inputs with Statsmodels
Specialized statsmodels graphics are used to create plots for prediction and interpretation, including:
- Plotting the fit of the linear model with respect to a selected input.
- Examining the training set predictions and residuals.
- Visualizing partial regression plots to account for multiple inputs.

## Conclusion
The case study demonstrates the importance of considering all inputs in a linear model and using specialized visualization tools to interpret the effects of multiple inputs on the output.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

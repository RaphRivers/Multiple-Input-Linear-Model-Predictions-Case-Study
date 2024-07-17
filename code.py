#!/usr/bin/env python
# coding: utf-8

# ## Working with Multiple Input Linear Models to Make Predictions
# In this case study I will use two input linear model features to interpret the results of predictions. 
# 
# The data used will be randomly generated  and fitted using the `statsmodels` interface to fit linear regression. I will interpret the results via the coefficient summaries and use specialized `statsmodels` graphics to extract insights into the behavior of the predictions. Let's get started by importing the required Python modules.
# 

# ## Import Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf


# ## Set Regression Coefficients for Linear Additive Features
# I will work with two inputs x1 and x2. These inputs will be used as additive features to model the relationships between each input and the average output, the `trend`. However, this model can be scaled to MANY inputs as applicable in real-world data. 
# 
# To generate the data we must specify the regression coefficients. Note you can use any value. For this case study, I will use  `b0 = -0.25`, `b1 = 1.95`, `b2 = 0.2`, where b is the unstandardized beta value representing the slope of the line between the predictor variable and the dependent variable.  

# In[2]:


# assign the additive features
b0 = -0.25
b1 = 1.95
b2 = 0.2


# ## Generate Random Input Data
# Assuming that the inputs have standard normal distributions with the `Mean = 0`, and the `Standard Deviation = 1`. Hence I will use a relatively small sample size (number of observations) of 35 data points and then generate a random number with a reproducible seed.

# In[3]:


# Set the number of observations
N = 35


# In[4]:


# Initialize random number generator and assign it to rg object
# Then Set the seed of 2100 for reproducibility
rg = np.random.default_rng(2100)


# In[5]:


# Call the NumPy random number generator for 2 input 
# The loc (Mean), Scale(Std), Size(Number of observations) 
x1 = rg.normal(loc=0, scale=1, size = N)
x2 = rg.normal(loc=0, scale=1, size = N)


# In[6]:


# Assign the inputs to the data frame using a dictionary
df_input = pd.DataFrame({'x1': x1, 'x2': x2})


# In[7]:


df_input.info()


# In[8]:


df_input.head()


# ### Calculate the TREND (Avg Output) assuming the LINEAR ADDITIVE FEATURES
# To calculate the average output or trend of the linear model I will use the linear model formula for continuous input that is linearly related to the TREND given the generated `input x1` and `input x2` and the 3 regression coefficients  `b0 = -0.25`, `b1 = 1.95`, `b2 = 0.2`. 
# 
# $$\mu = \beta_0 + \beta_1 \times x1 + \beta_2 \times x2 $$
# 
# This formula will allow us to add the effects (product) of the `input x1` to `input x2` for the output.
# 
# 

# In[9]:


# Make a hard copy of the df_input and assign it to df
df = df_input.copy()


# I will store the trend given the additive INPUTS values in a calculated column `trend`.  

# In[10]:


# Calculate the output and assign to a column [trend]
df['trend'] = b0 + b1 * df['x1'] + b2 * df['x2']
df.head()


# Note! that the regression linear model assumes that the output is normally distributed around its average, so the Gaussian distribution needs a standard deviation (sigma). As such I will set a sigma value and then will call one more Gaussian random number generator and set the `residual error` or remaining variation around the trend for the random output `y`. 

# In[11]:


# Set sigma
g_sigma = 0.85


# In[12]:


# Generate randam output Y using the avg output (trend) as mean, Scale(g_sigma), size will be the df sample size
df['y'] = rg.normal(loc=df.trend, scale=g_sigma, size=df.shape[0])


# In[13]:


df.info()


# In[14]:


df.head()


# Now that we have our modeled dataset let visualize the relationship between the inputs x1, x2 and output.

# ## Visualize the Relationships (Inputs and Output)
# Let's start with the relationship between input `x1` and output `y`. I will not use a line chart for this visualization because we are working with a randomly generated output that has unexplainable variation.

# In[15]:


# Examine the relationship between x1 and y
sns.relplot(data = df, x='x1', y ='y', kind='scatter' )
plt.show()


# The output `y` appears to be increasing as input `x1` increases. This indicates a positive relationship between the two.

# In[16]:


# Examine the relationship between x1 and y
sns.relplot(data = df, x='x2', y ='y', kind='scatter' )
plt.show()


# There is no difinitive relationship btween `x2` and `y`, so there is no relationship between the second input and the output. Next I will inclusde a bestfit line (trend line or regression line) to help further identify possible trends between the output and the inputs.

# In[17]:


sns.lmplot(data=df, x='x1', y='y')
plt.show()


# The regression line shown here assumes a linear relationship between the average output and the input x1. There seem to be a positive relationship between the input and output because of how narrow the confidence interval (CI) is. The CI here represents the uncertainty of the output and when the CI do not overlap that mean we are confident that the average output is different between the low median and high inputs. WHich mean that the average output is increasing as the input is increasing.

# In[18]:


# Examin the secong input
sns.lmplot(data=df, x='x2', y='y')
plt.show()


# For the second input x2 we can see that the CI is overlapping, therefore it is difficult to determine that the average output is changing as the input x2 is changing. Also the best fit line has a negative slope show that as the input `x2` is increasing the output `y` is decreasing. 

# ### Fit the Models and Summarize to Confirm the Visualization
# Let's start with the relationship between input x1 and output.

# In[19]:


# Fit a linear model btw the trend and x1 
x1_fit = smf.ols(data = df, formula='y ~ x1').fit()


# In[20]:


# Show the estimates
x1_fit.params


# From the estimates the slope multiplying x1 is positive `1.9`

# In[21]:


# show the standard error
x1_fit.bse


# The standard error is consistent

# In[22]:


# show the pvalues
x1_fit.pvalues


# In[23]:


x1_fit.pvalues < 0.05


# The pvalue is really small and is less than the common conventional 0.05 significance. Therefore x1 is a statistically significant positive slope.

# In[24]:


# Examine the 95% CI bounds
x1_fit.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'})


# The 95% confidence interval lower and upper bounds are both positive. So again we are confident that the relationship between input `x1` and the trend `y` is a positive relationship. 
# 
# Let's now verify the relationship between input x2 and the output. 

# In[25]:


x2_fit = smf.ols(data=df, formula='y ~ x2').fit()


# In[26]:


# Check the slope coefficient estimates given the data
x2_fit.params


# The slope on input x2 is negative

# In[27]:


# CHeck the standard error 
x2_fit.bse


# Standard error is consistent

# In[28]:


# Check the pvalues
x2_fit.pvalues


# In[29]:


x2_fit.pvalues < 0.05


# The pvalue is not less than 0.05, hence it is not statistically significant

# In[30]:


# Look at the confidence interval
x2_fit.conf_int().rename(columns={0: 'ci_lower', 1: 'ci_upper'})


# The confidence interval does not have the same sign the lower bound is negative and the upper bound is positive. Therefore we are not confident that the slope is definitely negative and thus not statistically significant. To examine this further, let's visualize the coefficient summaries using the approximation where the 95% CI is approximately 2 times the standard error. This ensures that our model works as intended even as we increase the standard error.

# ## Visualize the Coefficient Summaries

# In[31]:


# Define a function to show the coefficient summaries and error bar

def coeff_summary(model, figsize_used=(10, 4)):
    fig, ax = plt.subplots(figsize = figsize_used)

    ax.errorbar(y=model.params.index, x=model.params, xerr = 2 * model.bse, fmt='v', color='b', linewidth=2, ms=10)

    ax.axvline(x=0, linestyle='--', linewidth=2.5, color='r')

    ax.set_xlabel('Coefficient Value')

    plt.show()
        


# In[32]:


# Call the coefficient summary function given the input x1
coeff_summary(x1_fit)


# Looking at x1 you can see that the 95% CI does not contain 0 it is at approx 1.9. I can say that this is a statistically significant positive relationship. Let's call the function passing in input x2.

# In[33]:


# Call the coefficient summary function given the input x2
coeff_summary(x2_fit)


# Here you can see that the x2 in negative confirming that the 95% CI is not a statistically significant relationship. 
# 
# With that let's fit the features for both inputs

# ## Linear Additive Features with Multiple inputs

# In[34]:


# Fit both inputs x1 and x2 and assign to an object
x1x2_fit_add = smf.ols(data=df, formula='y ~ x1 + x2').fit() # Adding the inputs effects together


# In[35]:


# Check the estimates
x1x2_fit_add.params


# Now you can see that the `x2` slope is changed from a negative to a positive value because we have added the magnitude and effects of the model to the features. This illustrates that just because you fit a single input and have a negative slope doesn't mean that it is negative. However when you add the additives to the feature that negative slope can change. Because now the inputs are estimated together not separately. So if you have several features in your dataset, you cannot only fit your model for one input and conclude that it is negative you have to account for all the inputs.

# In[36]:


# Check the standard error
x1x2_fit_add.bse


# In[37]:


# check the pvalues
x1x2_fit_add.pvalues


# In[38]:


x1x2_fit_add.pvalues < 0.05


# Here only the x1 p-value is statistically significant even though the value of the x2 slope multiplying the x2 estimates is now positive. 

# In[39]:


# Check the confidence interval of both inputs
x1x2_fit_add.conf_int().rename(columns={0:'ci_lower', 1: 'ci_upper' })


# In[40]:


coeff_summary(x1x2_fit_add)


# We still have a negative slope on x2 which tells us that we are not 100% confident in that slope. As such it is tough to be able to identify the slope given the size of the data we have. However input x1 has the largest magnitude which makes it the most important input that we can use to make predictions.  
# 
# Let's now make predictions.

# ## Making Predictions
# Predictions involving multiple inputs require data frames that have all inputs used to fit the model. It is important to visualize the relationship between the output and the most important input. But how can you determine which input is the most important? When your inputs have the same magnitude and scale that is to say they are standardized. By construction of our inputs, they are because they both have `b0 = -0.25`, `b1 = 1.95`, `b2 = 0.2`. After fitting the input with the most magnitude and scale it eh most important input. Because the slope indicates how much the average output change for a one unit change of the input.

# In[41]:


# Let's sort the input absolute value according to th order of magnitude
np.abs(x1x2_fit_add.params).sort_values(ascending=False)


# Hence, input `x1` is the most important input that can be used for predictions. 
# 
# Now create a training set dataframe for amy values of x1 for a single value of x2

# ### Create a training set for x1

# In[42]:


# Create many values of x1 for 1 value of x2 into a df
df_predict_viz = pd.DataFrame({'x1': np.linspace(df.x1.min()-0.02, df.x1.max()+0.02, num=251)})
df_predict_viz


# In[43]:


# Add in x2 at a single constant value
df_predict_viz['x2'] = df.x2.mean() # this is the mean of the training set 
df_predict_viz


# The trianing set bounds of the most important input and the mean of the rest of the input are used to define the prediction grid. 
# 
# When we make a prediction we want to include columns,
# - the trend (Average output),
# - the uncertainty on the trend (Confidence Interval)
# - the uncertainty of the single measurment (Prediction Interval)

# In[44]:


# Prediction Step 1 - make a prediction using the training set for input x1 and x2
prediction_1 = x1x2_fit_add.get_prediction(df_predict_viz)


# In[45]:


# Prediction Step 2 - generate prediction set summary
prediction_x1x2_summary = prediction_1.summary_frame()


# In[46]:


# View prediction summary
prediction_x1x2_summary


# In[47]:


# Visualize the prediction using ribbons
fig, ax = plt.subplots()

# prediction interval PI
ax.fill_between(df_predict_viz.x1,  prediction_x1x2_summary.obs_ci_lower, prediction_x1x2_summary.obs_ci_upper,
               facecolor='blue', edgecolor='blue', alpha=0.5)

# Confidence Interval
ax.fill_between(df_predict_viz.x1,  prediction_x1x2_summary.mean_ci_lower, prediction_x1x2_summary.mean_ci_upper,
               facecolor='blue', edgecolor='blue', alpha=0.5)

# Trend line
ax.plot(df_predict_viz.x1, prediction_x1x2_summary['mean'], color='k', lw=2)

# Set labels
ax.set_xlabel('x1')
ax.set_ylabel('y')

# Show plot
plt.show()


# This PI chart is focused on the relationship between the prediction and the most important input `x1`. However, to examine the influence of other inputs we need to allow the `x2` input to also change. Hence the visualization df must allow for multiple combinations of continuous input so that we can see the whole relationship. The idea is to have an output for every input. For this we will need to create a grid of input combinations that are evenly spread, iterating in a sequence across x1 and x2 columns from the minimum to the maximum values of each input. Then assign this to a data frame and visualize. 

# In[48]:


# Create an input grid dataframe using a for loop 
x1x2_input_grid = pd.DataFrame([(x1, x2)
                          for x1 in np.linspace(df['x1'].min(), df['x1'].max(), num=101)
                          for x2 in np.linspace(df['x2'].min(), df['x2'].max(), num=9)],columns=['x1', 'x2'])


# In[49]:


# Preview the dataframe
x1x2_input_grid.info()


# In[50]:


x1x2_input_grid.nunique()


# So now we have evenly spaced data for both x1 and x2 inputs with x1 having 101 unique values and x2 nine. So that we can visualize the relationship between the most important input `x1` which we model to be statisticaaly significant and `x2` which wasn't. Now let's visualize this. 

# In[51]:


sns.relplot(data=x1x2_input_grid, x='x1', y='x2', kind='scatter')
plt.show()


# There are 101 more unique values of the x1 than the x2 that's why you see the dots here.  Now let's predict using the `input grid` but only returning the predicted trend (output). Why? Because dealing with multiple inputs in a prediction like this is very challenging to return the uncertainty. In my next case study, I will work with multiple input and output.  

# In[52]:


# Make a df hard copy and 
pred_grid_viz =  x1x2_input_grid.copy()


# In[53]:


# With the additive input fitted model use the .predict() to predict the trend and assign the output to a column 
pred_grid_viz['pred_trend'] = x1x2_fit_add.predict( pred_grid_viz )


# In[54]:


pred_grid_viz


# Now let's visualize the relationship using line colors to show how the prediction for the trend changes across input x2 across x1

# In[55]:


sns.relplot(data = pred_grid_viz, 
            x='x1', y='pred_trend', kind='line',
           hue='x2', palette='coolwarm', # colors to show x1 relationship with each x2
           estimator=None, units='x2') # disable units and estimates defining each line based on x2
plt.show()


# The separate line colors shows how the prediction for the trend changes across x1 for different value of x2. We see how the different values for the output increases as x1 increases and the input x2 has minimal impact on the trend as we previously saw. And we can prove that the slope that multiply x2 is larger than the slope tha multiplies x1.

# In[56]:


x1x2_fit_add.params # x1 is larger than x2 


# ### Why does this matter?
# The steps I performed above is to show how to manually interpret what is happening in the prediction of multiple input models. However, it far more easy to sue `statsmodel` function to help with visualizing linear model multiple inputs. Let's see that now...

# ## Visualizing Multiple Inputs Model with Statsmodels graphics

# In[57]:


# Import statsmodel api 
import statsmodels.api as sm


# We will use the statsmodel specialize figures to create plots for the prediction and interpret it

# In[58]:


# Create matplotlib figure and axis to use statsmodels plot
fig, ax = plt.subplots()
sm.graphics.plot_fit(x1x2_fit_add, 'x1', ax=ax) # Pass in fitted data and the associated axis
plt.show()


# This is saying to plot the fit of the linear model with respect to the selected input, in this case, x1. Note the input must be used in the fitted model formula.
# $$\mu = \beta_0 + \beta_1 \times x1 + \beta_2 \times x2 $$
# The result shows the fitted or the predictions of the training set. And that is critical when using the statsmodels specialize function. You will also notice that there are gaps shown at several points in the training set. This is because there are no observations at the predicted values around those points. Hence, the reason I created the prediction manually by making 251 continuous evenly spaced values between the x1.min() and x1.max.  
# 
# Note that this statsmodels visualization will only show you the predictions at the actual training points. The blue dots indicate the `observed outputs y`,  while the vertical line is the `prediction interval` PI which is the uncertainty of a single measurement. Thus we can see that the sigma is constant giving us an idea of where the predicted observation is located around the average. 
# 
# You will also notice the red diamond-shaped dots are not in a straight line whereas the manual prediction visualization was in a straight line. The reason is I assigned only one x2 value to the input. Meaning that you will not see a straight line when you have many inputs. Why we see a somewhat seemingly straight here is because x1 dominates the output-to-input relationship. To verify this let's visualize the training set fit for the x2 input.
# 

# In[59]:


fig, ax = plt.subplots()
sm.graphics.plot_fit(x1x2_fit_add, 'x2', ax=ax) # Pass in fitted data and the associated axis
plt.show()


# As we see earlier x2 does not show any reasonable predicted trend, that is because we are not looking at the relationship of x2 for one value of x1. Each one of those dots has a different value of x1 and x2. So we are actually getting the influence of the effect of both x1 and x2 here. We are just seeing what the training set for x2 looks like. However we can examine the trining set predictions and the residuals (errors) and the **partial regression plot in one figure.** 

# In[ ]:





# In[60]:


fig = plt.figure(figsize=(16,10))
sm.graphics.plot_regress_exog(x1x2_fit_add, 'x1', fig=fig) # Pass in fitted data to the regression exornogy
plt.show()


# This shows us the the output with respect to out input, the residuals (error), the regression plot, and the CCPR plot. Where the partial regression is the trend (avg output) given the value of the one input but accounts for the controls for the second input. Meaning that it is accounting for the fact that x2 is in the model. The CCPR (Component to Component regression plot) also accounts for the component of the second input for the output. And we can also look at x2 to visualize the influence of x1.

# In[61]:


fig = plt.figure(figsize=(15, 10))
sm.graphics.plot_regress_exog(x1x2_fit_add, 'x2', fig=fig)
plt.show()


# Notice the partial regression plot shows a positive trend for x2. This highlights the postive slope estimate for x2 when we account for x1. And that is what we had when we generated the data. 

# In[62]:


x1x2_fit_add.params


# In[63]:


b2 # x2 two slope


# In[64]:


x1_fit.params


# In[65]:


x2_fit.params


# ## In conclusion
# The positive regression above is a way to visualize how the trend or average output behaves, with response to 1 input but accounting for all other input in your model. So this shows that using lmplot is not intended to visualize your model because it only shows what is happening with one input at a time. While this statsmodels graphics account for the effects of multiple inputs on the most important input of your model. 
# 
# For working with multiple continuous inputs you need a model that is allowed to estimate their inputs and or control their effect.

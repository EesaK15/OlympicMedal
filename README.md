# OlympicMedal
This project utilizes machine learning to analyze historical data on Olympic medals won by nations, enabling it to make predictions on future medal outcomes.

To begin a machine learning project, we must follow the following steps: 
#### Step 1 - Form a Hypothesis
#### Step 2 - Find The Data
#### Step 3 - Reshape The Data
#### Step 4 - Clean The Data
#### Step 5 - Error Metric
#### Step 6 - Split The Data
#### Step 7 - Train a Model

You train the data on the training data and check its accuarcy on the test data
```
import pandas as pd
```
import pandas as pd allows you to import the Pandas library in Python and assign it the alias pd. 

```
teams = pd.read_csv(r'teams.csv')
teams.head()
```
Importing the 'teams.csv' file into the program 
```
# Take out extra columns

teams = teams[['team', 'country', 'year', 'athletes', 'age', 'prev_medals', 'medals']]
teams.head()
```
We have now taken the 'teams' variable and stored only the columns that we need suc has teams, country, year, atheltes, age, prev_medals and medals. We currently do not need the weight, age, hegiht or prev_3_medals. 

```
#checking correaltion between the medals column and other columns
teams.corr()['medals']
```
In this code, teams.corr()['medals'] calculates the correlation between the 'medals' column and all other columns in the DataFrame teams. It returns a Pandas Series that shows the correlation coefficients between the 'medals' column and each of the other columns, providing insights into how the 'medals' column is related to other features in the dataset. 

```
import seaborn as sns
sns.lmplot(x = 'athletes', y ='medals', data=teams, fit_reg= True, ci = None)
#shows a relation ship that the more atheltes we have, the more medals well win. We can see a linear relationship

teams.plot.hist(y ='medals')
```

The code uses the seaborn library to create a scatter plot with a linear regression line, showing the positive linear relationship between the number of athletes and the number of medals won in the DataFrame teams. The plot visually illustrates that an increase in the number of athletes is associated with a higher number of medals won.

### Data Cleaning
```
# Find rows that have missing values
teams[teams.isnull().any(axis=1)]
# Based off table below, Albania is NaN in the previous medals column because they did not participate in the olympics prior.
# We dont have medals for the year of 1988
```
In this code, teams[teams.isnull().any(axis=1)] filters the DataFrame teams to retrieve rows that have at least one missing value (NaN) in any column. It displays the rows where data is missing, and the comment below the code explains that the country "Albania" has a missing value in the 'medals' column because they did not participate in the Olympics prior to the given dataset, and therefore, there are no medals available for them for the year 1988.
```
teams = teams.dropna()
```
The dataset now does not have a null values.

We will now split the data; The data is time series, some rows should show up before some other rows, so we must split it in a way where we can  train with the past, and test with the future values. Well train with data from before 2012, and test it with 2012 data and 2016 data
```
train = teams[teams['year'] < 2012].copy()
test = teams[teams['year'] >= 2012].copy()
```
Splitting the data into 2 parts

### Training our model

```
from sklearn.linear_model import LinearRegression # this helps us train and make predicitions with a linear model
```
'from sklearn.linear_model import LinearRegression' imports the 'LinearRegression' class from the 'sklearn.linear_model module'. The LinearRegression class is a powerful tool that allows us to create and train a linear regression model, which is commonly used for predictive modeling in machine learning
```
reg = LinearRegression()

# we are gonna train our linear model to train the model
# this will predict the target column
predictors = ['athletes', 'prev_medals']
target = 'medals'

reg.fit(train[predictors], train[target])
```
In this code, reg = LinearRegression() creates an instance of the LinearRegression class, which represents a linear regression model.

Next, reg.fit(train[predictors], train[target]) is used to train the linear regression model. The training data consists of two predictor columns, namely 'athletes' and 'prev_medals', and the target column, which is 'medals'. The fit() method fits the model to the training data, allowing the linear regression model to learn the relationships between the predictor variables and the target variable. Once trained, the model can be used to make predictions on new data.

```
predictions = reg.predict(test[predictors])
predictions
```

reg.predict(test[predictors]) is used to make predictions using the previously trained linear regression model (reg) on the test dataset. The predict() method takes the test data as input and uses the learned relationships between the predictor variables ('athletes' and 'prev_medals') and the target variable ('medals') to generate predicted values for the target variable. 

```
test['predictions'] = predictions
```
```
test.loc[test['predictions']<0, 'predictions'] = 0

# index our test data frame, and find any rows where the pred columns is less than 0 and will replace it with 0. 
# so negatives are now 0
test.drop('predicitions', axis=1, inplace=True)
test['predictions'] = test['predictions'].round()
test
```

In this code, negative values in the 'predictions' column of the test DataFrame are replaced with 0, ensuring all predictions are non-negative. Then, the 'predictions' column is rounded to integers, and the 'predicitions' column (with a typo) is dropped from the DataFrame. These operations help in processing and refining the predictions for further analysis or presentation.
```
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(test['medals'], test['predictions'])
```
In this code, from sklearn.metrics import mean_absolute_error imports the mean_absolute_error function from scikit-learn (sklearn) library. The mean_absolute_error function is used to calculate the mean absolute error between the actual target values (test['medals']) and the predicted target values (test['predictions']). The resulting error variable holds the mean absolute error, which represents the average absolute difference between the actual and predicted values, providing a measure of the model's performance in making predictions.

```teams.describe()['medals']
```
In this code, teams.describe()['medals'] computes summary statistics for the 'medals' column in the DataFrame teams. The describe() function generates statistics such as count, mean, standard deviation, minimum, 25th percentile (1st quartile), median (2nd quartile), 75th percentile (3rd quartile), and maximum for the 'medals' column. This provides a quick overview of the distribution and central tendency of the 'medals' data in the DataFrame.

```
import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
```
The line error_ratio = error_ratio[np.isfinite(error_ratio)] filters the error_ratio array to keep only finite values. The np.isfinite() function creates a boolean array where each element is True if the corresponding element in error_ratio is finite (not NaN or infinity), and False otherwise. By indexing error_ratio with this boolean array, only the finite values are retained in the updated error_ratio array, removing any NaN or infinity values.

To improve, we may add more predictors, and try new models.   




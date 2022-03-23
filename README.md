#PROBLEM STATEMENT
Buying a home is one of the most expensive and stressful purchases a person can make.
What can we learn from studying home sale prices that can help you make a better decision as you buy your first home in Ames, Iowa?

#CONCLUSIONS AND RECOMMENDATIONS
Larger homes have higher sale prices.
Newer homes have higher sale prices.
Remodeling a home can increase its sale price.
Neighborhoods effect home prices:  A home in NorthRidge Heights will contribute more to a higher sale price than homes in other neighborhoods.  A home in the Edwards neighborhood will contribute the least to a home's sale price compared to all the other neighborhoods in Ames.


#DATA CLEANING AND EDA
Only two rows of data were dropped due to missing data.  Other features with large amounts of missing data were not considered for the analysis.

Please see code notebooks for more information on exploratory data analysis.

Outliers were not addressed - there didn't seem to be enough to take action on.

There is an entire "modeling diary" at the end of this document to describe the entire modeling process.

Here is a summary matrix of the models created and key measurements:

|Model Type| Features Included|R Squared|MAE  |  MSE|
|Baseline|Average Home Price|-0.00167|53997.23|5131236795.21|
|Simple Linear Regression|Garage Area, Total Bsmt SF|0.5424|36280.51|2344237361.11|
|Lasso Model 1|See Below|0.6697|26255.44|2158173486.19|
|Lasso Model 2|See Below|0.8874|18201.28|808623144.94|
|Ridge Model 1|See Below|0.8353|20569.71|978267812.73|
|Ridge Model 2|See Below|0.8108|17958.98|1177288517.80|
|Linear Regression Model 1|Overall Qual, Neighborhood|0.7480|28320.91|1821540668.09|
|Linear Regression Model 2|See Below|0.8722|18619.68|761773729.18|
|Linear Regression Model 3|See Below|0.8657|19555.79|914179160.45|
|Linear Regression Model 4|See Below|0.8786|18791.51|798726699.05|
|Linear Regression Model 5|See Below|0.8214|22009.88|982918626.74|

Note:  Based on R Squared, my best model was the Lasso Model 2 with an R Squared of .8874.  During my presentation, I talked about a Ridge Model being the best.  Unfortunately, I couldn't find the code for that as I was organizing my notebooks.  It is entirely possible that I was mistaken, and it was actually the Lasso model that was the best! 

However, after doing all this analysis, I still have other models that I would like to try because I know that I have multicollinearity issues in these models here!

Lasso Model 1 Features: Gr Liv Area, Garage Cars,
Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,
TotRms AbvGrd, Fireplaces, BsmtFin SF 1,
Open Porch SF, Wood Deck SF, Lot Area

Lasso Model 2 Features: Overall Qual,Gr Liv Area, Garage Cars,
Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,
TotRms AbvGrd, Fireplaces, BsmtFin SF 1,Neighborhood,
Open Porch SF, Wood Deck SF, Lot Area

Ridge Model 1 Features: 'Gr Liv Area', 'Garage Cars',
'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Year Remod/Add', 'Full Bath',
'TotRms AbvGrd', 'Fireplaces', 'BsmtFin SF 1',
'Open Porch SF', 'Wood Deck SF', 'Lot Area'

Ridge Model 2 Features: 'Overall Qual','Gr Liv Area', 'Garage Cars',
'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Year Remod/Add', 'Full Bath',
'TotRms AbvGrd', 'Fireplaces', 'BsmtFin SF 1','Neighborhood',
'Open Porch SF', 'Wood Deck SF', 'Lot Area'

Linear Regression Model #2 Features: 'Overall Qual','Gr Liv Area', 'Garage Cars',
'Total Bsmt SF', '1st Flr SF', 'Year Built', 'Year Remod/Add', 'Full Bath',
'TotRms AbvGrd', 'Fireplaces', 'BsmtFin SF 1','Neighborhood',
'Open Porch SF', 'Wood Deck SF', 'Lot Area'

Linear Regression Model #3 Features:  'Overall Qual','Neighborhood',
'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', '1st Flr SF',
'Year Built', 'Year Remod/Add', 'Full Bath'

Linear Regression Model #4 Features: 'Overall Qual','Neighborhood','Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', '1st Flr SF','Year Built', 'Year Remod/Add'

Linear Regression Model #5 Features: 'Neighborhood',
'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', '1st Flr SF',
'Year Built', 'Year Remod/Add'

#MY DATA SCIENCE DIARY
Note:  This is long, but it describes my thought process and growth on this amazing experience of completing this project!

#Notebook: p2-nb1-eda-and-simple-lin-reg-model-sarah-sturgeon
#Exploratory Data Process
After I read in the train data, I used .describe to look at my data and see where I have large chunks of missing data.  Right away I see that these features have missing data:  Lot Frontage and Mas Vnr Area.  There may be more features missing data since I can’t see them all in this output.
Next I want to look at correlations of all the features.  I first look at correlations among all the features and SalePrice, which is overwhelming, so I then move to only looking at the correlations between SalePrice and the features.  I am looking for a subset of the 70 features that might be most predictive of SalePrice to build my model.
From this correlation of features with SalePrice, I choose the following features, all with correlations to SalePrice of .54 or higher:  Overall Qual, Gr Liv Area, Garage Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add and Full Bath.
I reran the correlation matrix across all chosen features and SalePrice, but I don’t look closely at it.  This bites me later and causes me to run more Linear Regression models than I need to.  Things that I should have noticed here:  Full Bath is highly correlated with Gr Liv Area at .62.  Gr Liv Area has a higher correlation with SalePrice, so that is the variable to keep in the model (which I ultimately did).  I also knew that Garage Area and Garage Cars have a correlation of .89 to each other and only .65 to SalePrice, so I will drop one of these.  Garage Cars is easier for most people to picture (a 1-car vs a 2-car garage), so I keep this feature in my final Linear Regression model.  If I had looked at this correlation matrix longer, I would have noticed that there are two other sets of features that are highly correlated with each other.  Total Bsmt SF and 1st Flr SF have a correlation of .81 with each other versus .63 and .62 respectively with SalePrice.  I should keep Total Bsmt SF in my model because it is more highly correlated with SalePrice.  The next pair of features that are highly correlated are Year Built and Year Remod/Add with a correlation of .63.  Their correlations with SalePrice are .57 and .55, respectively, so I should keep Year Built in my model and drop Year Remod/Add.
When I check this subset of features for missing data, I find that there are at most 3 instances of missing data, and I am very comfortable dropping 3 observations to avoid dealing with missing data in my model.  Ultimately, I only need to drop 2 observations.
I create a baseline model here using the average SalePrice as my predicted Sale Price.  This turns out to be an awful model!
First Kaggle Model Submission
I decide to use only two features and focus on creating a simple Linear Regression model to do a model submission to Kaggle.  I was going to use the top 3 most correlated features to SalePrice, but I become obsessed with Overall Qual and decide it is really a categorical feature rather than a numeric one, and I’m not ready to tackle OneHotEncoder yet.  So my simple Linear Regression model submission to Kaggle becomes a model based on only Garage Area and Total Basement SF.  And yes, for my presentation I choose the Garage Cars feature rather than the Garage Area feature.  At this point, I’m just really concerned about completing my Kaggle submission.

Linear Regression model submission to Kaggle csv file:  './datasets/test.csv'

#Notebook: p2-nb2-lasso-model-1-sarah-sturgeon
Exploratory Data Process (again!)
I look at value counts of the nine features I chose based on high correlation values to SalePrice and also look through the data dictionary to see if there are any obvious or important ones I have missed.
#Lasso Model #1
I have a ton of features for my Lasso model:  Overall Qual, Gr Liv Area, Garage Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath, Garage Yr Blt, Mas Vnr Area, TotRms AbvGrd, Fireplaces, BsmtFin SF 1, Lot Frontage, Open Porch SF, Wood Deck SF, and Lot Area.  This is like a “kitchen sink” model – I want to see what comes out of it.
I look at pairplots of all of these features, subdivided into groups of 5 with SalePrice.
Ultimately, I decide to go with these variables in my Lasso model:  Gr Liv Area,  Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,  TotRms AbvGrd, Fireplaces, BsmtFin SF 1,  Open Porch SF, Wood Deck SF, and Lot Area.  I use Standard Scaler to standardize the data because I have square feet and simple measure like # of cars in a garage.  I also offer Lasso parameters of .01, .1, 1, 10, and 100.  I use GridSearch to determine the best Lasso alpha, and it selects an alpha of 100.
I also submit this model to Kaggle: ./datasets/las_predictions.csv

#Notebook: p2-nb3-lasso-model-2-sarah-sturgeon
#Lasso Model #2
I am itching to add the two categorical features to my Lasso model: Overall Qual and Neighborhood.
So here are the features I used in this second Lasso model: Overall Qual (exploded into categorical features), Gr Liv Area,  Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,  TotRms AbvGrd, Fireplaces, BsmtFin SF 1,  Open Porch SF, Wood Deck SF, and Lot Area.
I used OneHotEncoder to create my categorical features. I then joined this in my pipeline with StandardScaler and Lasso.  
I passed coefficients of .005, .01, .1, 1, 100 and 200.  I needed to add 200 because my model wasn’t converging using my range to 100.  
I then set up a GridSearchCV to determine which Lasso alpha I should use, and it chose the Lasso of 200.  I am guessing it is because I had all the additional categorical variables from the OneHotEncoder process. 
I did not submit this model to Kaggle because it didn’t beat the Ridge model.

#Notebook: p2-nb4-ridge-model-1-sarah-sturgeon
#Ridge Model #1
I want to use the Ridge method to build a model.
Here are the features I used in this Ridge model: Gr Liv Area,  Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,  TotRms AbvGrd, Fireplaces, BsmtFin SF 1,  Open Porch SF, Wood Deck SF, and Lot Area.
I created a pipeline with PolynomialFeatures, StandardScaler and Ridge.  
I pass coefficients of .01, .1, 1, 10 and 100.  
I then set up a GridSearchCV to determine which Ridge alpha I should use, and it chose the Ridge alpha of 100.  
I also submitted my model predictions to Kaggle: './datasets/rdg_predictions.csv'

#Notebook: p2-nb5-ridge-model-2-sarah-sturgeon
#Ridge Model #2
I wanted to add the two categorical features to my Ridge model: Overall Qual and Neighborhood.
So here are the features I used in this second Ridge model: Overall Qual (exploded into categorical features), Gr Liv Area,  Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath,  TotRms AbvGrd, Fireplaces, BsmtFin SF 1,  Open Porch SF, Wood Deck SF, and Lot Area.
I used OneHotEncoder to create my categorical features and used the “drop first” option for the categorical features to reduce multicollinearity. I then joined this in my pipeline with StandardScaler and Ridge.  
I passed coefficients of .01, .1, 1, 10, and 100.  
I then set up a GridSearchCV to determine which Ridge alpha I should use, and it chose the Ridge alpha of 100.  
I did not submit this model to Kaggle because it didn’t beat a previous model.

#Notebook: p2-nb6-linear-regression-2catonly-model-1-sarah-sturgeon
#Linear Regression Model #1: 2 Categorical Features Only 
I only used the Neighborhood and the Overall Qual features in this model.  I wanted to se if they were strong enough to stand on their own and they were not.

#Notebook: p2-nb7-linear-regression-lots-feats-model-2-sarah-sturgeon
#Linear Regression Model #2: Lots of Features Model
This was my “kitchen sink” model, where I threw a bunch of features in to see how well it would perform.
The features I included were: Overall Qual (categorical), Gr Liv Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, Full Bath, TotRms AbvGrd, Fireplaces, BsmtFin SF1, Neighborhood (categorical), Open Porch SF, Wood Deck SF, Lot Area.
When looking at the model coefficients, Overall Qual didn’t seem right.  I knew that there were features in here that were not particularly highly correlated with Sale Price, so I decided to simplify the model and pull a bunch of these “extra” features out to see if Overall Qual would make more sense.

#Notebook: p2-nb8-linear-regression-simpl-feats-model-3-sarah-sturgeon
#Linear Regression Model #3: Simplified Features Model
I decided to keep the features that had been highly correlated with Sale Price to see if the coefficients for Overall Qual would improve.
Here are the features that I used in this model:  Overall Qual (categorical), Neighborhood (categorical), Gr Liv Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, Year Remod/Add, and Full Bath.
Overall Qual coefficients still didn’t make sense and Full Bath had a negative coefficient which didn’t make sense.  Knowing that Overall Qual was more highly correlated with Sale Price led me to drop Full Bath since I couldn’t explain the negative coefficient (it was because of multicollinearity with Gr Liv Area so it was a good decision to drop Full Bath).  

#Notebook: p2-nb9-linear-regression-nofullbath-model-4-sarah-sturgeon
#Linear Regression Model 4: No Full Bath Feature
I dropped Full Bath from the model hoping that Overall Qual coefficients would make more sense.
Here are the features that I used in this model:  Overall Qual (categorical), Neighborhood (categorical), Gr Liv Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built, and Year Remod/Add.
Sadly, Overall Qual coefficients didn’t improve, so I decided to drop this from my model.

#Notebook: p2-nb10-linear-regression-final-prez-model-5-sarah-sturgeon
#Linear Regression Presentation Model
The features I used in this model were: Neighborhood (created as categorical through OneHotEncoder), Gr Liv Area, Garage Cars, Total Bsmt SF, 1st Flr SF, Year Built and Year Remod/Add.
I chose the Linear Regression model technique over Ridge and Lasso because it would be simpler and more intuitive to explain during the presentation.  This model didn’t perform better than the Ridge models.
I believe that this final model can be improved by dropping 1st Flr SF because it is highly correlated to Total Bsmt SF.  I also believe that Year Remod/Add should be dropped because it is highly correlated with Year Built.  Both Total Bsmt SF and Year Built have highly correlations with SalePrice which is why they should be kept.


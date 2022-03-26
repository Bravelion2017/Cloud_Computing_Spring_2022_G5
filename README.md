
# PREDICTIVE MODELING FOR CHANCE OF ADMISSION

The Code main.py shows the EDA Analysis, as well as the results of two machine learning algorithms based on the chance of admission data set. The research's aim is to build a regression model that best predicts the chance of admission for graduate students and obtain feature importance.

Code Order: The code is being executed from top to bottom 

Data set structure:

Independent Variable -- X  
GRE Score
TOEFL Score
SOP  (Statement of Purpose)
LOR (Letter of Recommendation)
CGPA(Cumulative GPA)
University Rating
Research 

Target/ Dependant Variable -- Y
Chance of Admission


Machine Learning algorithms:

Random Forest Regression
Linear Regression

Requirements:
Before running this  main.py file, you need the following installed on your python environment:
Pip install numpy
Pip install pandas
Pip install matplotlib
Pip install seaborn
Pip install scikit-learn


Note: Using Anaconda environment saves the stress of the above apart from those in asterisk and the link.



Description of files:


Code main.py is the python file that contains all the code for the models.


All the files need to be located in the same directory

Technical considerations:


The application also uses graphviz-2.38, to be user to have it installed in the computer

The directory that uses this application for graphviz-2.38 is:

'C:\\Program Files (x86)\\graphviz-2.38\\release\\bin'
If you have it installed in another path, change this path at the top of the Main.py file.


Description of the application:


The structure of the application is as follows


E.D.A Analysis
Distribution : This option presents a frequency histogram. It shows the distribution of  Y variable (Change of Admission)
Features vs Admit Chance : This option presents a scatter plot that shows the relation of each feature in the datasets with Change of Admission. A line can be drawn optionally to represent the tendency of the data.
Heatmap : This option presents a correlation plot for all the features in the dataset. The features can be added or deleted from the plot. Each time that a modification is made the button Create Plot should be pressed.
Boxplot: This option shows a box plot which shows the range and possible outliers of the selected features




ML MODELS

Cross Validation:
The section for cross validation displays first hand test scores on the data set with several regression models. This gives us a view on what model works best for our data set.

Random Forest Regression, Linear Regression


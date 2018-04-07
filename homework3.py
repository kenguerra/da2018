# Step 1
# Download the file default.csv from Session 3 on Canvas Files
# Step 2
# Add these Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as sma
import seaborn as sns

# Step 3
# Add these lines to account for an incompatible package in statsmodels
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# Step 4
# Add this convenient function that helps draw a line given slope/intercept

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    axes.set_autoscale_on(False)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

# LET'S DO LINEAR REGRESSION
d = pd.read_csv("smarket.csv")

print(d.describe())

#sns.set(style="ticks", color_codes=True)
#g = sns.pairplot(d)
#plt.show()
#print("\nPairwise correlation coefficients", d.corr())

# Logistic Regression
d['DirectionUp'] = d['Direction'].map({'Up': 1, 'Down': 0})
res = sm.glm(formula="DirectionUp ~ Lag1+Lag2+Lag3+Lag4+Lag5",data=d,family=sma.families.Binomial()).fit()
print(res.summary())



# get only year 2005
smpl = d[ d["Year"] == 2005 ]
trn = d.drop(smpl.index)
print(trn)

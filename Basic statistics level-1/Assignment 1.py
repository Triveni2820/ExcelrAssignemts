# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:58:32 2023

@author: user
"""
'''
Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     comment about the values / draw inferences, for the given dataset
-	For Points,Score,Weigh>
Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.
Use Q7.csv file '''
import pandas as pd 
import numpy as np
df = pd.read_csv("C:/Users/user/Downloads/Q7.csv")
df

df.head()
df.tail()
#mean
df.mean()

#median
df.median()

#mode
#Weigh

df.Weigh.mode()
df.Score.mode()

#mode
df.Points.mode()

#standard variance
df.std()

#variance:
df.var()

#Range:
df.describe()
df_point = df.Points.max() - df.Points.min()
df_weigh = df.Weigh.max() - df.Weigh.min()
df_score = df.Score.max() - df.Score.min()

#=============================================================================
'''
Q8) Calculate Expected Value for the problem below
a)	The weights (X) of patients at a clinic (in pounds), are
108, 110, 123, 134, 135, 145, 167, 187, 199
Assume one of the patients is chosen at random. What is the Expected Value of the Weight of that patient? '''

x = np.array([108, 110, 123, 134, 135, 145, 167, 187, 199])

weights = x.mean() 
weights      


#=============================================================================
'''
Q9) Calculate Skewness, Kurtosis & draw inferences on the following data
      Cars speed and distance 
Use Q9_a.csv'''

import scipy.stats as skew
import scipy.stats as kurtosis
df =pd.read_csv("C:/Users/user/Downloads/Q9_a.csv")   
df    

df['speed'].skew()
df['speed'].kurtosis()

df['dist'].skew()
df['dist'].kurtosis()   

#drawing the  
df_new = df.loc[ :, ['speed','dist']]
df_new.boxplot()
df_new.hist()

#=============================================================================
'''
SP and Weight(WT)
Use Q9_b.csv'''

df = pd.read_csv("C:/Users/user/Downloads/Q9_b.csv")
df  
df.skew()
df.kurtosis()

df_new = df.loc[ :, ['SP','WT']]
df_new.boxplot()
df_new.hist()

#=============================================================================
'''
Q11)  Suppose we want to estimate the average weight of an adult male in
    Mexico. We draw a random sample of 2,000 men from a population of 
    3,000,000 men and weigh them. We find that the average person in our 
    sample weighs 200 pounds, and the standard deviation of the sample is 30 
    pounds. Calculate 94%,98%,96% confidence interval?'''
import pandas as pd
import numpy as np
from scipy import stats
#Average weight person in Mexico with 94%=0.96 confidence interval
#alpha =6%
df_ci = stats.norm.interval(0.94,200,30/np.sqrt(2000))
print("Average weight person in Mexico with 94%=0.94 confidence interval:",df_ci)
#output:Average weight person in Mexico with 94%=0.94 confidence interval(198.738325292158, 201.261674707842)

#Average weight person in Mexico with 98%=0.98 confidence interval
#alpha =2%
df_ci = stats.norm.interval(0.98,200,30/np.sqrt(2000))
    
print("Average weight person in Mexico with 98%=0.98 confidence interval:",df_ci)
#output:Average weight person in Mexico with 98%=0.98 confidence interval(198.43943840429978, 201.56056159570022)

#Average weight person in Mexico with 96%=0.96 confidence interval
#alpha =4%
df_ci = stats.norm.interval(0.96,200,30/np.sqrt(2000))
print("Average weight person in Mexico with 96%=0.96 confidence interval:",df_ci)


#======================================================================
'''
Q12)  Below are the scores obtained by a student in tests 
34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
1)	Find mean, median, variance, standard deviation.
2)	What can we say about the student marks? '''

import pandas as pd 
import numpy as np
x = np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])


x.mean()
np.median(x)
x.std() 
x.var()

#====================================================================================
'''
Q 20) Calculate probability from the given dataset for the below cases

Data _set: Cars.csv
Calculate the probability of MPG  of Cars for the below cases.
       MPG <- Cars$MPG
a.	P(MPG>38)
b.	P(MPG<40)
c.  P (20<MPG<50)'''

import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("C:/Users/user/Downloads/Cars (1).csv")
df
mpg =df["MPG"]
mpg
#a.	P(MPG>38)
p=38
n=mpg.mean()
d=mpg.std()
p_38=1-stats.norm.cdf(p,n,d)
print("the probability MPG>38%:",p_38)
#b.P(MPG<40)
p=40
p_40=stats.norm.cdf(p,n,d)
print("the probability MPG>40%:",p_40)
#c.P (20<MPG<50)
p_20_50=stats.norm.cdf(50,n,d) - stats.norm.cdf(20,n,d)
print("The probabilty 20<MPG<50:",p_20_50)

#==============================================================================
'''
Q 21) Check whether the data follows normal distribution
a)	Check whether the MPG of Cars follows Normal Distribution 
        Dataset: Cars.csv'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/user/Downloads/Cars (1).csv")
df

df['MPG'].hist()

sns.distplot(df['MPG'])
plt.grid(True)
plt.show()

#=============================================================================
'''
b) Check whether the Adipose Tissue(AT) and Waist Circumference(Waist)  from
    wc-at data set  follows Normal Distribution 
       Dataset: wc-at.csv'''
      
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/user/Downloads/wc-at.csv")
df.shape
df
df.mean()
df.median()
df.mode()
sns.distplot(df['Waist'])
plt.show()


sns.distplot(df['AT'])
plt.show()

sns.boxplot(df['AT'])
plt.show()
# mean> median, right whisker is larger than left whisker, data is positively skewed.
sns.boxplot(df['Waist'])
plt.show()
# mean> median, both the whisker are of same lenght, median is slightly shifted towards left. Data is fairly symetrical

#=========================================================================================
'''
Q 22) Calculate the Z scores of  90% confidence interval,94% confidence
    interval, 60% confidence interval '''
    
import numpy as np
from scipy import stats
from scipy.stats import norm 

''' 90% 
#0.90 = c
#alpha = 1 - c= 1- 0.90 = 0.10 
# alpha/2 = 0.5
# 1-0.5 = 0.95'''
# Z-score of 90% confidence interval 
z_score = stats.norm.ppf(0.95)
print("Z-score of 90% confidence interval :",z_score)
# Z-score of 94% confidence interval
z_score = stats.norm.ppf(0.97)
print("Z-score of 94% confidence interval :",z_score)
# Z-score of 60% confidence interval
z_score = stats.norm.ppf(0.8)
print("Z-score of 60% confidence interval :",z_score)

#=============================================================================================
'''
Calculate the t scores of 95% confidence interval, 96% confidence     interval,
99% confidence interval for sample size of 25'''

import numpy as np
from scipy import stats
from scipy.stats import norm

'''
1+0.95/2 
0.97,0.98,0.995
n=25'''
df = 24

t_score=stats.t.ppf(0.97,df) #here df is same
print("t-score of 95% confidence interval:",t_score)


t_score=stats.t.ppf(0.98,df) #here df is same
print("t-score of 96% confidence interval:",t_score)

t_score=stats.t.ppf(0.995,df) #here df is same
print("t-score of 99% confidence interval:",t_score)

#=======================================================================================
'''
  Q 24)   A Government  company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days
Hint:  
   rcode   pt(tscore,df)  
 df  degrees of freedom'''
 
import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import norm
stats.t.cdf(-0.47,17)


  


    



    
    

    
       

        







    




  


    





    


    


# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:09:23 2023

@author: user
"""

import pandas as pd
import numpy as np
df = pd.read_csv("C:/Users/user/Downloads/Costomer+OrderForm.csv")
df

df.shape
df.head()
df.tail()


df.describe()

"""Phillippines   Indonesia       Malta       India
count           300         300         300         300
unique            2           2           2           2
top      Error Free  Error Free  Error Free  Error Free
freq            271         267         269         280"""

df.info()

"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 300 entries, 0 to 299
Data columns (total 4 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   Phillippines  300 non-null    object
 1   Indonesia     300 non-null    object
 2   Malta         300 non-null    object
 3   India         300 non-null    object
dtypes: object(4)
memory usage: 9.5+ KB"""

#making Error Free and Defective a table 
df_updated = np.array([[271,267,269,280],[29,33,31,20]])

#importing scipy package
import scipy.stats as stats

#here using chi-contingency 
stats.chi2_contingency(df_updated)

#chisqure_value:3.858960685820355,
pvalue=0.2771020991233135
#DregeeofFreedom:3

if pvalue < 0.05:
    print("ho is rejected and h1 is accepted")
else:
    print("h1 is rejected and ho is accepeted ")
#p-value > 0.05,h1 is rejected and ho is accepeted 
# Hence the Customer order form defective % doesn't varies across the given countries
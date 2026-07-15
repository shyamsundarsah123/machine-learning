import pandas as pd
import numpy as np
from numpy.linalg import matrix_rank
#A1
df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')

df = df.iloc[:, :5]
X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
rank = np.linalg.matrix_rank(X)


xinv = np.linalg.pinv(X)
y = df['Payment (Rs)'].values.reshape(-1, 1)

#A2

cost = xinv @ y
print(f"Rank = {rank}")
print(df)
print(f"Pseudoinverse of the matrix X: {xinv}")
print(f"Cost vector: {cost}")

for i in range(len(y)):
    if(y[i]<200):
        print(f"Customer {i+1} is Poor")
    else:
        print(f"Customer {i+1} is Rich")
#A3


df2 = pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')


df2 = df2.iloc[:, :9]
print((df2))
D = df2['Price'].values.reshape(-1, 1)
sum2 = 0 
for i in range(len(D)):
    sum2 += D[i]
mean = sum2/len(D)
print(f"Mean of the stock price = {mean}")
var = np.var(D)
variance = sum((x - mean) ** 2 for x in D) / len(D)
print(f"Variance of the stock price = {var}")
print(f"Variance of the stock price = {variance}")
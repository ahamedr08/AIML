# Importing necessary libraries
import numpy as np
import pandas as pd

# Reading the dataset from a CSV file
data = pd.read_csv('play_tennis.csv')

# Displaying the first few rows of the dataset
data.head()

# Dropping the 'day' column as it is not needed for classification
data.drop(columns=['day'], inplace=True)

# Displaying the modified dataset
data

# Counting the occurrences of each target class ('play' column)
data['play'].value_counts()

# Calculating the prior probabilities for each target class
py = 9/14  # P(Play = Yes)
pn = 5/14  # P(Play = No)

# Conditional probabilities for the 'outlook' attribute
pd.crosstab(data['outlook'], data['play'])
pon = 0
prn = 2/5
psn = 3/5
poy = 4/9
pry = 3/9
psy = 2/9

# Conditional probabilities for the 'temp' attribute
pd.crosstab(data['temp'], data['play'])
pcoolno = 1/5
photno = 2/5
pmildno = 2/5
pcoolyes = 3/9
photyes = 2/9
pmildyes = 4/9

# Conditional probabilities for the 'humidity' attribute
pd.crosstab(data['humidity'], data['play'])
phighno = 4/5
pnormalno = 1/5
phighyes = 3/9
pnormalyes = 6/9

# Conditional probabilities for the 'wind' attribute
pd.crosstab(data['wind'], data['play'])
pstrongno = 3/5
pweakno = 2/5
psrongyes = 3/9
pweakyes = 6/9

# Calculating the probability of playing tennis and not playing tennis
pyes = py * psy * photyes * phighyes * pweakyes
pno = pn * psn * photno * phighno * pweakno

# Displaying the probabilities
print("P(Play = Yes) =", pyes)
print("P(Play = No) =", pno)

# Making the classification decision based on probabilities
if pyes > pno:
    print("Play Tennis")
else:
    print("Do Not Play Tennis")
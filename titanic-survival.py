import pandas as pd

# Reading Csv File To draw Conclusions and Analyse Data
train = pd.read_csv("train.csv")

# Printing First Five Data From Csv File
print(train.head())

# Setting Initials To Null
print(train.isnull().sum())

# Train Shape mean its rows and columns
print("Train Shape:",train.shape)

x = train.isnull().sum()
print(x)

# Droping Rows 
drop_col = x[x>(35/100*train.shape[0])]
print(drop_col.index)
train.drop(drop_col.index,axis=1,inplace=True)

# Filling Blank Values To Mean Values
train.fillna(train.mean(), inplace = True)
print(train.isnull().sum())

# Filling Blank Values in Embarked Column To 'S'
print('Before Filling empty rows :',train['Embarked'].describe())
train['Embarked'].fillna('S', inplace = True)
print('After Filling empty rows :',train['Embarked'].describe())

# Depicting Co-relation B/w Columns
print(train.corr())

# Storing Sibsp and parch in new field Family size
train['Family size'] = train['SibSp']+train['Parch']
train.drop(['SibSp','Parch'],axis=1,inplace=True)

#Predicting Survival if alone
train['Alone'] = [0 if train['Family size'][i]>0 else 1 for i in train.index]
print(train.groupby(['Alone'])['Survived'].mean())
print(train[['Alone','Fare']].corr())

#Predicting Survival on basis of Sex
train['Sex'] = [0 if train['Sex'][i]=='male' else 1 for i in train.index]
print(train.groupby(['Sex'])['Survived'].mean())

#Predicting Survival on basis of Embarked Point -- 'C','S','Q'
print(train.groupby(['Embarked'])['Survived'].mean())

''' Conclusions
1.Female passengers were prioritized over men.
2.People with high class or rich people have higher survival rate than others. the hierarichy might be followed while saving.
3.Passengers with families have higher chances of survival.
4.Passengers who boarded the ship at cherbourg , survived more.
'''
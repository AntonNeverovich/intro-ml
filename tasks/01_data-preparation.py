import pandas
from utils import raw_data_preparation as rdp

data = pandas.read_csv("data/_ea07570741a3ec966e284208f588e50e_titanic.csv", index_col='PassengerId')

print("Task #1. Quantity Males and Females")
print(data['Sex'].value_counts(0))

print("\nTask #2. Part of Survived")
surv = data['Survived'].value_counts(1) * 100
print(surv.round(2))

print("\nTask #3. Part of Passengers in First class")
passed = data['Pclass'].value_counts(1) * 100
print(passed.round(2))

print("\nTask #4. Passengers' age: mean and median")
print(data['Age'].mean().round(2))
print(data['Age'].median().round(2))

print("\nTask #5. Pearson correlation")
print(data['SibSp'].corr(data['Parch']).round(2))

print("\nTask #6. Most popular female name")
list = data[data['Sex'] == 'female']['Name']
result_set = []

for row in list:
    temp = row.split()
    for i in temp:
        result_set.append(i)


rdp.cleaning_data1(result_set, ',', '(', ')', '\"')
answ = pandas.DataFrame(result_set, columns=['first_name'])
answ = answ[answ['first_name'] != 'Mrs.']
answ = answ[answ['first_name'] != 'Miss.']
print(answ['first_name'].value_counts())
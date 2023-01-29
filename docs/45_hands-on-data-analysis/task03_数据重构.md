```python
import numpy as np
import pandas as pd
text_left_up = pd.read_csv("data/train-left-up.csv")
text_left_down = pd.read_csv("data/train-left-down.csv")
text_right_up = pd.read_csv("data/train-right-up.csv")
text_right_down = pd.read_csv("data/train-right-down.csv")
```

#  第二章：数据重构


### 2.4 数据的合并

#### 2.4.1 任务一：将data文件夹里面的所有数据都载入，观察数据的之间的关系


```python
#left与right的数据是横向分割，up与down是纵向分割的数据
text_left_up.head()
text_right_up.head()
text_left_down.head()
text_right_down.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 18723</td>
      <td>10.500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>45.0</td>
      <td>1</td>
      <td>1</td>
      <td>F.C.C. 13529</td>
      <td>26.250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>345769</td>
      <td>9.500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>25.0</td>
      <td>1</td>
      <td>0</td>
      <td>347076</td>
      <td>7.775</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>230434</td>
      <td>13.000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.2：任务二：使用concat方法：将数据train-left-up.csv和train-right-up.csv横向合并为一张表，并保存这张表为result_up


```python
list_up = [text_left_up,text_right_up]
result_up = pd.concat(list_up,axis=1)
result_up.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.3 任务三：使用concat方法：将train-left-down和train-right-down横向合并为一张表，并保存这张表为result_down。然后将上边的result_up和result_down纵向合并为result。


```python

list_down=[text_left_down,text_right_down]
result_down = pd.concat(list_down,axis=1)
result = pd.concat([result_up,result_down])
result.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.4 任务四：使用DataFrame自带的方法join方法和append：完成任务二和任务三的任务


```python

resul_up = text_left_up.join(text_right_up)
result_down = text_left_down.join(text_right_down)
result = result_up.append(result_down)
result.head()
```

    /var/folders/nm/41yw47353z12bzs83s_xfcl80000gn/T/ipykernel_8269/631931502.py:3: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      result = result_up.append(result_down)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.5 任务五：使用Panads的merge方法和DataFrame的append方法：完成任务二和任务三的任务


```python
result_up = pd.merge(text_left_up,text_right_up,left_index=True,right_index=True)
result_down = pd.merge(text_left_down,text_right_down,left_index=True,right_index=True)
result = resul_up.append(result_down)
result.head()
```

    /var/folders/nm/41yw47353z12bzs83s_xfcl80000gn/T/ipykernel_8269/2576447027.py:3: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      result = resul_up.append(result_down)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



【思考】对比merge、join以及concat的方法的不同以及相同。思考一下在任务四和任务五的情况下，为什么都要求使用DataFrame的append方法，如何只要求使用merge或者join可不可以完成任务四和任务五呢？

#### 2.4.6 任务六：完成的数据保存为result.csv


```python
result.to_csv('result.csv')
```

### 2.5 换一种角度看数据

#### 2.5.1 任务一：将我们的数据变为Series类型的数据


```python
# 将完整的数据加载出来
text = pd.read_csv('result.csv')
text.head()
# stack函数会新开辟一个新的维度，将原始维度在这个新的维度拼接起来
unit_result=text.stack().head(20)
unit_result
```




    0  Unnamed: 0                                                     0
       PassengerId                                                    1
       Survived                                                       0
       Pclass                                                         3
       Name                                     Braund, Mr. Owen Harris
       Sex                                                         male
       Age                                                         22.0
       SibSp                                                          1
       Parch                                                          0
       Ticket                                                 A/5 21171
       Fare                                                        7.25
       Embarked                                                       S
    1  Unnamed: 0                                                     1
       PassengerId                                                    2
       Survived                                                       1
       Pclass                                                         1
       Name           Cumings, Mrs. John Bradley (Florence Briggs Th...
       Sex                                                       female
       Age                                                         38.0
       SibSp                                                          1
    dtype: object




```python
unit_result.to_csv('unit_result.csv')
test = pd.read_csv('unit_result.csv')
test.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Unnamed: 0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>PassengerId</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Survived</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Pclass</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Name</td>
      <td>Braund, Mr. Owen Harris</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>Sex</td>
      <td>male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>Age</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>SibSp</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>Parch</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>Ticket</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>Fare</td>
      <td>7.25</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>Embarked</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>Unnamed: 0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>PassengerId</td>
      <td>2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>Survived</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 第一部分：数据聚合与运算

### 2.6 数据运用

#### 2.6.1 任务一：通过教材《Python for Data Analysis》P303、Google or anything来学习了解GroupBy机制


```python
# df.groupby('key1')[['data2']]
# df.groupby('key1')['data1']
# df[data1'].groupby(df['key1'])
# df[[data2']].groupby(df['key1'])
```

#### 2.4.2：任务二：计算泰坦尼克号男性与女性的平均票价


```python

means = text['Fare'].groupby(text['Sex']).mean()
means
```




    Sex
    female    44.479818
    male      25.523893
    Name: Fare, dtype: float64



在了解GroupBy机制之后，运用这个机制完成一系列的操作，来达到我们的目的。

下面通过几个任务来熟悉GroupBy机制。

#### 2.4.3：任务三：统计泰坦尼克号中男女的存活人数


```python

text.groupby(['Sex'])["Survived"].sum()
```




    Sex
    female    233
    male      109
    Name: Survived, dtype: int64




```python
one = text.groupby(['Pclass','Sex']).count()[['PassengerId']].reset_index()
one = one.pivot_table(index='Pclass',columns='Sex')
one.columns = [item[1] for item in one.columns]
one['female_ratio'] = one.apply(lambda x: x.female/x.sum(),axis=1 )
one['male_ratio'] = one.apply(lambda x: x.male/x.sum(),axis=1 )

```


```python
one
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female</th>
      <th>male</th>
      <th>female_ratio</th>
      <th>male_ratio</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>94</td>
      <td>122</td>
      <td>0.435185</td>
      <td>0.563679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76</td>
      <td>108</td>
      <td>0.413043</td>
      <td>0.585642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>144</td>
      <td>347</td>
      <td>0.293279</td>
      <td>0.706299</td>
    </tr>
  </tbody>
</table>
</div>



```python

survived_sex = text['Survived'].groupby(text['Sex']).sum()
survived_sex.head()
```


    Sex
    female    233
    male      109
    Name: Survived, dtype: int64



```python
survived_sex = text.groupby('Sex').sum()[['Survived']]
survived_sex.head()
```

    /var/folders/nm/41yw47353z12bzs83s_xfcl80000gn/T/ipykernel_8269/3212473813.py:1: FutureWarning: The default value of numeric_only in DataFrameGroupBy.sum is deprecated. In a future version, numeric_only will default to False. Either specify numeric_only or select only columns which should be valid for the function.
      survived_sex = text.groupby('Sex').sum()[['Survived']]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.4：任务四：计算客舱不同等级的存活人数


```python
survived_pclass = text['Survived'].groupby(text['Pclass'])
survived_pclass.sum()
```


    Pclass
    1    136
    2     87
    3    119
    Name: Survived, dtype: int64


### 思考
1. 在男女比例为`577：314`的情况下，最后存活下来的比例却是反比`109:233`.
2. 女性虽然跟男性一样分布于不同等级的`pclass`，但等级越高的女性比例越高,男性缺与之相反.
3. 在不同位置调用函数的参数会得到`series`(一维）或者`dataframe`（二维）结果.
4. `pclass`的等级越高，存活的比例越高.
5. 还可以通过更多的数据分析比较得到信息，比如获救人数中女性的存活跟`Pclass`是否有关联.

【思考】从任务二到任务三中，这些运算可以通过agg()函数来同时计算。并且可以使用rename函数修改列名。你可以按照提示写出这个过程吗？


#### 2.4.5：任务五：统计在不同等级的票中的不同年龄的船票花费的平均值


```python
text.groupby('Sex').agg({'Fare': 'mean', 'Pclass': 'count'}).rename(columns=
                            {'Fare': 'mean_fare', 'Pclass': 'count_pclass'})

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean_fare</th>
      <th>count_pclass</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>44.479818</td>
      <td>314</td>
    </tr>
    <tr>
      <th>male</th>
      <td>25.523893</td>
      <td>577</td>
    </tr>
  </tbody>
</table>
</div>



#### 2.4.6：任务六：将任务二和任务三的数据合并，并保存到sex_fare_survived.csv


```python
result = pd.merge(means,survived_sex,on='Sex')
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fare</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>44.479818</td>
      <td>233</td>
    </tr>
    <tr>
      <th>male</th>
      <td>25.523893</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>




```python
result.to_csv('sex_fare_survived.csv')
```

#### 2.4.7：任务七：得出不同年龄的总的存活人数，然后找出存活人数最多的年龄段，最后计算存活人数最高的存活率（存活人数/总人数）



```python
text.groupby(['Pclass','Age'])['Fare'].mean().head()
```


    Pclass  Age  
    1       0.92     151.5500
            2.00     151.5500
            4.00      81.8583
            11.00    120.0000
            14.00    120.0000
    Name: Fare, dtype: float64



```python

survived_age = text.groupby('Age')['Survived'].sum()
survived_age.head()
```


    Age
    0.42    1
    0.67    1
    0.75    2
    0.83    2
    0.92    1
    Name: Survived, dtype: int64



```python

survived_age[survived_age.values==survived_age.max()]

```


    Age
    24.0    15
    Name: Survived, dtype: int64



```python
#计算存活总人数，对数字进行字符串输出
_sum = text['Survived'].sum()
print("sum of person:"+str(_sum))
precent =survived_age.max()/_sum
print("最大存活率："+str(precent))
```

    sum of person:342
    最大存活率：0.043859649122807015


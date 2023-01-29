## 第一章：数据载入及初步观察

### 1.1 载入数据
数据集下载 https://www.kaggle.com/c/titanic/overview

#### 1.1.1 任务一：导入numpy和pandas


```python
import numpy as np
import pandas as pd
import os
```

#### 1.1.2 任务二：载入数据
(1) 使用相对路径载入数据  
(2) 使用绝对路径载入数据


```python
#查看当前工作目录
os.getcwd()
```




    '/Users/apple/GitHub/datawhale-team-learning/notebooks/45_hands-on-data-analysis/chapter1'




```python
#查找绝对路径
road= os.path.abspath("train.csv")
road
```




    '/Users/apple/GitHub/datawhale-team-learning/notebooks/45_hands-on-data-analysis/chapter1/train.csv'




```python
#使pd.read_csv()与pd.read_table()运行结果相同，设定分隔符参数。
df = pd.read_table('data/train.csv',sep = ',')
df.head(3)
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
  </tbody>
</table>
</div>




```python
#读取绝对路径过长时，不易阅读，使用路径叠加
project_dir = '/Users/apple/GitHub/datawhale-team-learning/'
data_file = 'notebooks/45_hands-on-data-analysis/chapter1/data/train.csv'
data_dir = os.path.join(project_dir,data_file)
df = pd.read_csv(data_dir)
df.head()
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



#### 了解一下`.tsv`和`.csv`的不同，如何加载这两个数据集？  
- csv和tsv都是文本文件；xlsx是二进制文件，xlsx只能用Excel打开
- csv(Comma-separated values)默认`逗号（comma)`为分隔符（delimiter）
- tsv(Tab-separated values)默认`tab制表符('\t')`为分隔符（delimiter）

#### 1.1.3 任务三：每1000行为一个数据模块，逐块读取


```python
pd.read_csv('data/train.csv').shape
```




    (891, 12)




```python
#逐块读取：为避免整体读取造成内存运行速度降低，同时占用内存空间（计算机资源）
#chunksize:Return TextFileReader object for iteration
#每100条逐块读取对此次数据更有意义
chunker = pd.read_csv('data/train.csv', chunksize=100)
print(type(chunker))
print(type(df))
#每一个row都是dataframe类型的
for row in chunker:
      print (row)
      i = i+1
print(i)
```

    <class 'pandas.io.parsers.readers.TextFileReader'>
    <class 'pandas.core.frame.DataFrame'>
        PassengerId  Survived  Pclass  \
    0             1         0       3   
    1             2         1       1   
    2             3         1       3   
    3             4         1       1   
    4             5         0       3   
    ..          ...       ...     ...   
    95           96         0       3   
    96           97         0       1   
    97           98         1       1   
    98           99         1       2   
    99          100         0       2   
    
                                                     Name     Sex   Age  SibSp  \
    0                             Braund, Mr. Owen Harris    male  22.0      1   
    1   Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                              Heikkinen, Miss. Laina  female  26.0      0   
    3        Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                            Allen, Mr. William Henry    male  35.0      0   
    ..                                                ...     ...   ...    ...   
    95                        Shorney, Mr. Charles Joseph    male   NaN      0   
    96                          Goldschmidt, Mr. George B    male  71.0      0   
    97                    Greenfield, Mr. William Bertram    male  23.0      0   
    98               Doling, Mrs. John T (Ada Julia Bone)  female  34.0      0   
    99                                  Kantor, Mr. Sinai    male  34.0      1   
    
        Parch            Ticket     Fare    Cabin Embarked  
    0       0         A/5 21171   7.2500      NaN        S  
    1       0          PC 17599  71.2833      C85        C  
    2       0  STON/O2. 3101282   7.9250      NaN        S  
    3       0            113803  53.1000     C123        S  
    4       0            373450   8.0500      NaN        S  
    ..    ...               ...      ...      ...      ...  
    95      0            374910   8.0500      NaN        S  
    96      0          PC 17754  34.6542       A5        C  
    97      1          PC 17759  63.3583  D10 D12        C  
    98      1            231919  23.0000      NaN        S  
    99      0            244367  26.0000      NaN        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass                                    Name  \
    100          101         0       3                 Petranec, Miss. Matilda   
    101          102         0       3        Petroff, Mr. Pastcho ("Pentcho")   
    102          103         0       1               White, Mr. Richard Frasar   
    103          104         0       3              Johansson, Mr. Gustaf Joel   
    104          105         0       3          Gustafsson, Mr. Anders Vilhelm   
    ..           ...       ...     ...                                     ...   
    195          196         1       1                    Lurette, Miss. Elise   
    196          197         0       3                     Mernagh, Mr. Robert   
    197          198         0       3        Olsen, Mr. Karl Siegwart Andreas   
    198          199         1       3        Madigan, Miss. Margaret "Maggie"   
    199          200         0       2  Yrois, Miss. Henriette ("Mrs Harbeck")   
    
            Sex   Age  SibSp  Parch    Ticket      Fare Cabin Embarked  
    100  female  28.0      0      0    349245    7.8958   NaN        S  
    101    male   NaN      0      0    349215    7.8958   NaN        S  
    102    male  21.0      0      1     35281   77.2875   D26        S  
    103    male  33.0      0      0      7540    8.6542   NaN        S  
    104    male  37.0      2      0   3101276    7.9250   NaN        S  
    ..      ...   ...    ...    ...       ...       ...   ...      ...  
    195  female  58.0      0      0  PC 17569  146.5208   B80        C  
    196    male   NaN      0      0    368703    7.7500   NaN        Q  
    197    male  42.0      0      1      4579    8.4042   NaN        S  
    198  female   NaN      0      0    370370    7.7500   NaN        Q  
    199  female  24.0      0      0    248747   13.0000   NaN        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    200          201         0       3   
    201          202         0       3   
    202          203         0       3   
    203          204         0       3   
    204          205         1       3   
    ..           ...       ...     ...   
    295          296         0       1   
    296          297         0       3   
    297          298         0       1   
    298          299         1       1   
    299          300         1       1   
    
                                                    Name     Sex   Age  SibSp  \
    200                   Vande Walle, Mr. Nestor Cyriel    male  28.0      0   
    201                              Sage, Mr. Frederick    male   NaN      8   
    202                       Johanson, Mr. Jakob Alfred    male  34.0      0   
    203                             Youseff, Mr. Gerious    male  45.5      0   
    204                         Cohen, Mr. Gurshon "Gus"    male  18.0      0   
    ..                                               ...     ...   ...    ...   
    295                                Lewy, Mr. Ervin G    male   NaN      0   
    296                               Hanna, Mr. Mansour    male  23.5      0   
    297                     Allison, Miss. Helen Loraine  female   2.0      1   
    298                            Saalfeld, Mr. Adolphe    male   NaN      0   
    299  Baxter, Mrs. James (Helene DeLaudeniere Chaput)  female  50.0      0   
    
         Parch    Ticket      Fare    Cabin Embarked  
    200      0    345770    9.5000      NaN        S  
    201      2  CA. 2343   69.5500      NaN        S  
    202      0   3101264    6.4958      NaN        S  
    203      0      2628    7.2250      NaN        C  
    204      0  A/5 3540    8.0500      NaN        S  
    ..     ...       ...       ...      ...      ...  
    295      0  PC 17612   27.7208      NaN        C  
    296      0      2693    7.2292      NaN        C  
    297      2    113781  151.5500  C22 C26        S  
    298      0     19988   30.5000     C106        S  
    299      1  PC 17558  247.5208  B58 B60        C  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass                                      Name  \
    300          301         1       3  Kelly, Miss. Anna Katherine "Annie Kate"   
    301          302         1       3                        McCoy, Mr. Bernard   
    302          303         0       3           Johnson, Mr. William Cahoone Jr   
    303          304         1       2                       Keane, Miss. Nora A   
    304          305         0       3         Williams, Mr. Howard Hugh "Harry"   
    ..           ...       ...     ...                                       ...   
    395          396         0       3                       Johansson, Mr. Erik   
    396          397         0       3                       Olsson, Miss. Elina   
    397          398         0       2                   McKane, Mr. Peter David   
    398          399         0       2                          Pain, Dr. Alfred   
    399          400         1       2          Trout, Mrs. William H (Jessie L)   
    
            Sex   Age  SibSp  Parch    Ticket     Fare Cabin Embarked  
    300  female   NaN      0      0      9234   7.7500   NaN        Q  
    301    male   NaN      2      0    367226  23.2500   NaN        Q  
    302    male  19.0      0      0      LINE   0.0000   NaN        S  
    303  female   NaN      0      0    226593  12.3500  E101        Q  
    304    male   NaN      0      0  A/5 2466   8.0500   NaN        S  
    ..      ...   ...    ...    ...       ...      ...   ...      ...  
    395    male  22.0      0      0    350052   7.7958   NaN        S  
    396  female  31.0      0      0    350407   7.8542   NaN        S  
    397    male  46.0      0      0     28403  26.0000   NaN        S  
    398    male  23.0      0      0    244278  10.5000   NaN        S  
    399  female  28.0      0      0    240929  12.6500   NaN        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    400          401         1       3   
    401          402         0       3   
    402          403         0       3   
    403          404         0       3   
    404          405         0       3   
    ..           ...       ...     ...   
    495          496         0       3   
    496          497         1       1   
    497          498         0       3   
    498          499         0       1   
    499          500         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    400                               Niskanen, Mr. Juha    male  39.0      0   
    401                                  Adams, Mr. John    male  26.0      0   
    402                         Jussila, Miss. Mari Aina  female  21.0      1   
    403                   Hakkarainen, Mr. Pekka Pietari    male  28.0      1   
    404                          Oreskovic, Miss. Marija  female  20.0      0   
    ..                                               ...     ...   ...    ...   
    495                            Yousseff, Mr. Gerious    male   NaN      0   
    496                   Eustis, Miss. Elizabeth Mussey  female  54.0      1   
    497                  Shellard, Mr. Frederick William    male   NaN      0   
    498  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female  25.0      1   
    499                               Svensson, Mr. Olof    male  24.0      0   
    
         Parch             Ticket      Fare    Cabin Embarked  
    400      0  STON/O 2. 3101289    7.9250      NaN        S  
    401      0             341826    8.0500      NaN        S  
    402      0               4137    9.8250      NaN        S  
    403      0   STON/O2. 3101279   15.8500      NaN        S  
    404      0             315096    8.6625      NaN        S  
    ..     ...                ...       ...      ...      ...  
    495      0               2627   14.4583      NaN        C  
    496      0              36947   78.2667      D20        C  
    497      0          C.A. 6212   15.1000      NaN        S  
    498      2             113781  151.5500  C22 C26        S  
    499      0             350035    7.7958      NaN        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    500          501         0       3   
    501          502         0       3   
    502          503         0       3   
    503          504         0       3   
    504          505         1       1   
    ..           ...       ...     ...   
    595          596         0       3   
    596          597         1       2   
    597          598         0       3   
    598          599         0       3   
    599          600         1       1   
    
                                                 Name     Sex   Age  SibSp  Parch  \
    500                              Calic, Mr. Petar    male  17.0      0      0   
    501                           Canavan, Miss. Mary  female  21.0      0      0   
    502                O'Sullivan, Miss. Bridget Mary  female   NaN      0      0   
    503                Laitinen, Miss. Kristina Sofia  female  37.0      0      0   
    504                         Maioni, Miss. Roberta  female  16.0      0      0   
    ..                                            ...     ...   ...    ...    ...   
    595                   Van Impe, Mr. Jean Baptiste    male  36.0      1      1   
    596                    Leitch, Miss. Jessie Wills  female   NaN      0      0   
    597                           Johnson, Mr. Alfred    male  49.0      0      0   
    598                             Boulos, Mr. Hanna    male   NaN      0      0   
    599  Duff Gordon, Sir. Cosmo Edmund ("Mr Morgan")    male  49.0      1      0   
    
           Ticket     Fare Cabin Embarked  
    500    315086   8.6625   NaN        S  
    501    364846   7.7500   NaN        Q  
    502    330909   7.6292   NaN        Q  
    503      4135   9.5875   NaN        S  
    504    110152  86.5000   B79        S  
    ..        ...      ...   ...      ...  
    595    345773  24.1500   NaN        S  
    596    248727  33.0000   NaN        S  
    597      LINE   0.0000   NaN        S  
    598      2664   7.2250   NaN        C  
    599  PC 17485  56.9292   A20        C  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    600          601         1       2   
    601          602         0       3   
    602          603         0       1   
    603          604         0       3   
    604          605         1       1   
    ..           ...       ...     ...   
    695          696         0       2   
    696          697         0       3   
    697          698         1       3   
    698          699         0       1   
    699          700         0       3   
    
                                                      Name     Sex   Age  SibSp  \
    600  Jacobsohn, Mrs. Sidney Samuel (Amy Frances Chr...  female  24.0      2   
    601                               Slabenoff, Mr. Petco    male   NaN      0   
    602                          Harrington, Mr. Charles H    male   NaN      0   
    603                          Torber, Mr. Ernst William    male  44.0      0   
    604                    Homer, Mr. Harry ("Mr E Haven")    male  35.0      0   
    ..                                                 ...     ...   ...    ...   
    695                         Chapman, Mr. Charles Henry    male  52.0      0   
    696                                   Kelly, Mr. James    male  44.0      0   
    697                   Mullens, Miss. Katherine "Katie"  female   NaN      0   
    698                           Thayer, Mr. John Borland    male  49.0      1   
    699           Humblen, Mr. Adolf Mathias Nicolai Olsen    male  42.0      0   
    
         Parch  Ticket      Fare  Cabin Embarked  
    600      1  243847   27.0000    NaN        S  
    601      0  349214    7.8958    NaN        S  
    602      0  113796   42.4000    NaN        S  
    603      0  364511    8.0500    NaN        S  
    604      0  111426   26.5500    NaN        C  
    ..     ...     ...       ...    ...      ...  
    695      0  248731   13.5000    NaN        S  
    696      0  363592    8.0500    NaN        S  
    697      0   35852    7.7333    NaN        Q  
    698      1   17421  110.8833    C68        C  
    699      0  348121    7.6500  F G63        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    700          701         1       1   
    701          702         1       1   
    702          703         0       3   
    703          704         0       3   
    704          705         0       3   
    ..           ...       ...     ...   
    795          796         0       2   
    796          797         1       1   
    797          798         1       3   
    798          799         0       3   
    799          800         0       3   
    
                                                      Name     Sex   Age  SibSp  \
    700  Astor, Mrs. John Jacob (Madeleine Talmadge Force)  female  18.0      1   
    701                   Silverthorne, Mr. Spencer Victor    male  35.0      0   
    702                              Barbara, Miss. Saiide  female  18.0      0   
    703                              Gallagher, Mr. Martin    male  25.0      0   
    704                            Hansen, Mr. Henrik Juul    male  26.0      1   
    ..                                                 ...     ...   ...    ...   
    795                                 Otter, Mr. Richard    male  39.0      0   
    796                        Leader, Dr. Alice (Farnham)  female  49.0      0   
    797                                   Osman, Mrs. Mara  female  31.0      0   
    798                       Ibrahim Shawah, Mr. Yousseff    male  30.0      0   
    799  Van Impe, Mrs. Jean Baptiste (Rosalie Paula Go...  female  30.0      1   
    
         Parch    Ticket      Fare    Cabin Embarked  
    700      0  PC 17757  227.5250  C62 C64        C  
    701      0  PC 17475   26.2875      E24        S  
    702      1      2691   14.4542      NaN        C  
    703      0     36864    7.7417      NaN        Q  
    704      0    350025    7.8542      NaN        S  
    ..     ...       ...       ...      ...      ...  
    795      0     28213   13.0000      NaN        S  
    796      0     17465   25.9292      D17        S  
    797      0    349244    8.6833      NaN        S  
    798      0      2685    7.2292      NaN        C  
    799      1    345773   24.1500      NaN        S  
    
    [100 rows x 12 columns]
         PassengerId  Survived  Pclass  \
    800          801         0       2   
    801          802         1       2   
    802          803         1       1   
    803          804         1       3   
    804          805         1       3   
    ..           ...       ...     ...   
    886          887         0       2   
    887          888         1       1   
    888          889         0       3   
    889          890         1       1   
    890          891         0       3   
    
                                                Name     Sex    Age  SibSp  Parch  \
    800                         Ponesell, Mr. Martin    male  34.00      0      0   
    801  Collyer, Mrs. Harvey (Charlotte Annie Tate)  female  31.00      1      1   
    802          Carter, Master. William Thornton II    male  11.00      1      2   
    803              Thomas, Master. Assad Alexander    male   0.42      0      1   
    804                      Hedman, Mr. Oskar Arvid    male  27.00      0      0   
    ..                                           ...     ...    ...    ...    ...   
    886                        Montvila, Rev. Juozas    male  27.00      0      0   
    887                 Graham, Miss. Margaret Edith  female  19.00      0      0   
    888     Johnston, Miss. Catherine Helen "Carrie"  female    NaN      1      2   
    889                        Behr, Mr. Karl Howell    male  26.00      0      0   
    890                          Dooley, Mr. Patrick    male  32.00      0      0   
    
             Ticket      Fare    Cabin Embarked  
    800      250647   13.0000      NaN        S  
    801  C.A. 31921   26.2500      NaN        S  
    802      113760  120.0000  B96 B98        S  
    803        2625    8.5167      NaN        C  
    804      347089    6.9750      NaN        S  
    ..          ...       ...      ...      ...  
    886      211536   13.0000      NaN        S  
    887      112053   30.0000      B42        S  
    888  W./C. 6607   23.4500      NaN        S  
    889      111369   30.0000     C148        C  
    890      370376    7.7500      NaN        Q  
    
    [91 rows x 12 columns]
    18



```python
#chunker(iter类型)用for进行读取数据后成为空容器,编译chunker.get_chunk会出现stopiteration。
#nrows参数是chunksize的数据模块的范围限制，默认值为None。
chunker.nrows==None
```




    True




```python
reader = pd.read_csv( 'data/train.csv',iterator=True)
reader.get_chunk(5)
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
      <td>22</td>
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
      <td>38</td>
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
      <td>26</td>
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
      <td>35</td>
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
      <td>35</td>
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



#### 1.1.4 任务四：将表头改成中文，索引改为乘客ID [对于某些英文资料，我们可以通过翻译来更直观的熟悉我们的数据]
PassengerId => 乘客ID  
Survived    => 是否幸存   
Pclass      => 乘客等级(1/2/3等舱位)  
Name        => 乘客姓名  
Sex         => 性别                 
Age         => 年龄                 
SibSp       => 堂兄弟/妹个数  
Parch       => 父母与小孩个数  
Ticket      => 船票信息             
Fare        => 票价                
Cabin       => 客舱                
Embarked    => 登船港口             


```python

df = pd.read_csv(data_dir, names=['乘客ID','是否幸存','仓位等级','姓名','性别','年龄','兄弟姐妹个数','父母子女个数','船票信息','票价','客舱','登船港口'],index_col='乘客ID',header=0)
df.head()

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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
    <tr>
      <th>乘客ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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




```python
names=['乘客ID','是否幸存','仓位等级','姓名','性别','年龄','兄弟姐妹个数','父母子女个数','船票信息','票价','客舱','登船港口']
train_rename = pd.read_csv(data_dir)
train_rename.columns = names
train_rename.head()
train_rename.set_index('乘客ID')
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
    <tr>
      <th>乘客ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 11 columns</p>
</div>



### 1.2 初步观察
导入数据后，你可能要对数据的整体结构和样例进行概览，比如说，数据大小、有多少列，各列都是什么格式的，是否包含null等

#### 1.2.1 任务一：查看数据的基本信息


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 1 to 891
    Data columns (total 11 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   是否幸存    891 non-null    int64  
     1   仓位等级    891 non-null    int64  
     2   姓名      891 non-null    object 
     3   性别      891 non-null    object 
     4   年龄      714 non-null    float64
     5   兄弟姐妹个数  891 non-null    int64  
     6   父母子女个数  891 non-null    int64  
     7   船票信息    891 non-null    object 
     8   票价      891 non-null    float64
     9   客舱      204 non-null    object 
     10  登船港口    889 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 83.5+ KB


【提示】有多个函数可以这样做，你可以做一下总结

#### 1.2.2 任务二：观察表格前10行的数据和后15行的数据


```python
df.head(10)
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
    <tr>
      <th>乘客ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>5</th>
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
    <tr>
      <th>6</th>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



```python
df.tail(5)
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
    <tr>
      <th>乘客ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>887</th>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.00</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.00</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>


#### 1.2.4 任务三：判断数据是否为空，为空的地方返回True，其余地方返回False


```python
df.isnull().head()
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
    <tr>
      <th>乘客ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
#描述性统计信息
df.describe()
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
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>票价</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>


【思考】对于一个数据，还可以从哪些方面来观察？找找答案，这个将对下面的数据分析有很大的帮助


```python
#列行数
df.shape
```


    (891, 11)


### 1.3 保存数据

#### 1.3.1 任务一：将你加载并做出改变的数据，在工作目录下保存为一个新文件train_chinese.csv


```python
df.to_csv('train_chinese.csv')
#数据的加载以及入门，接下来就要接触数据本身的运算，我们将主要掌握numpy和pandas在工作和项目场景的运用。
```

### 1.4 知道你的数据叫什么
我们学习pandas的基础操作，那么上一节通过pandas加载之后的数据，其数据类型是什么呢？

#### 1.4.1 任务一：pandas中有两个数据类型DateFrame和Series，通过查找简单了解他们。然后自己写一个关于这两个数据类型的小例子🌰[开放题]


```python
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
example_1 = pd.Series(sdata)
example_1
```


    Ohio      35000
    Texas     71000
    Oregon    16000
    Utah       5000
    dtype: int64


- series是一维带标签数组，左边是index，右边是values
- dataframe是二维，dataframe既有行索引也有列索引


```python
fdata = pd.DataFrame(np.arange(10).reshape(2,5),index=list("ab"),columns=list("qwert"))
print(fdata)
```

       q  w  e  r  t
    a  0  1  2  3  4
    b  5  6  7  8  9



```python
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
example_2 = pd.DataFrame(data)
example_2
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
      <th>state</th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ohio</td>
      <td>2000</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ohio</td>
      <td>2001</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ohio</td>
      <td>2002</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nevada</td>
      <td>2001</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nevada</td>
      <td>2002</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Nevada</td>
      <td>2003</td>
      <td>3.2</td>
    </tr>
  </tbody>
</table>
</div>


#### 1.4.2 任务二：根据上节课的方法载入"train.csv"文件



```python
df = pd.read_csv('data/train.csv')
```

也可以加载上一节课保存的"train_chinese.csv"文件。通过翻译版train_chinese.csv熟悉了这个数据集，然后我们对trian.csv来进行操作
#### 1.4.3 任务三：查看DataFrame数据的每列的名称


```python
#列名信息
print(df.columns)
print(df.ndim)
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')
    2


#### 1.4.4任务四：查看"Cabin"这列的所有值[有多种方法]


```python
df.Cabin.head()
```




    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: Cabin, dtype: object




```python
df['Cabin'].head()
```


    0     NaN
    1     C85
    2     NaN
    3    C123
    4     NaN
    Name: Cabin, dtype: object


#### 1.4.5 任务五：加载文件"test_1.csv"，然后对比"train.csv"，看看有哪些多出的列，然后将多出的列删除
经过我们的观察发现一个测试集test_1.csv有一列是多余的，我们需要将这个多余的列删去


```python

test_1 = pd.read_csv('data/test_1.csv')
del test_1['a']
test_1.head(3)
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
      <td>0</td>
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
      <td>1</td>
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
      <td>2</td>
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
  </tbody>
</table>
</div>



#### 1.4.6 任务六： 将['PassengerId','Name','Age','Ticket']这几个列元素隐藏，只观察其他几个列元素


```python
#使用pd.drop()函数，需指明为纵轴
#参数inplace = True将改变原始数据，inpalce = False时不改变原数据，故可以隐藏列元素
df.drop(['PassengerId','Name','Age','Ticket'],axis=1).head(3)
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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


### 1.5 筛选的逻辑

表格数据中，最重要的一个功能就是要具有可筛选的能力，选出我所需要的信息，丢弃无用的信息。

#### 1.5.1 任务一： 我们以"Age"为筛选条件，显示年龄在10岁以下的乘客信息。


```python
df[df["Age"]<10].head(3)
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
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.075</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.700</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Master. Eugene</td>
      <td>male</td>
      <td>2.0</td>
      <td>4</td>
      <td>1</td>
      <td>382652</td>
      <td>29.125</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.2 任务二： 以"Age"为条件，将年龄在10岁以上和50岁以下的乘客信息显示出来，并将这个数据命名为midage


```python

midage = df[(df["Age"]>10)& (df["Age"]<50)]
midage.head(3)
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
  </tbody>
</table>
</div>



#### 1.5.3 任务三：将midage的数据中第100行的"Pclass"和"Sex"的数据显示出来


```python
midage.loc[[100],['Pclass','Sex']]
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
      <th>Pclass</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
#进行二次筛选，需要额外的处理，使用drop参数设置去掉原索引（默认false）
midage = midage.reset_index(drop=True)
midage.head(3)
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
      <th>index</th>
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
      <td>0</td>
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
      <td>1</td>
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
      <td>2</td>
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
  </tbody>
</table>
</div>



#### 1.5.4 任务四：使用loc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
#Selection by Label,通过标签索引行数据
midage.loc[[100,105,108],['Pclass','Name','Sex']] 
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
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
      <td>male</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
      <td>male</td>
    </tr>
    <tr>
      <th>108</th>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.5.5 任务五：使用iloc方法将midage的数据中第100，105，108行的"Pclass"，"Name"和"Sex"的数据显示出来


```python
#Selection by Position,只接受整数作为参数
midage.iloc[[100,105,108],[2,3,4]]

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
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>2</td>
      <td>Byles, Rev. Thomas Roussel Davids</td>
    </tr>
    <tr>
      <th>105</th>
      <td>0</td>
      <td>3</td>
      <td>Cribb, Mr. John Hatfield</td>
    </tr>
    <tr>
      <th>108</th>
      <td>0</td>
      <td>3</td>
      <td>Calic, Mr. Jovo</td>
    </tr>
  </tbody>
</table>
</div>




```python
#载入之前保存的train_chinese.csv数据，关于泰坦尼克号的任务，我们就使用这个数据
text = pd.read_csv('data/train_chinese.csv')
```

### 1.6 了解你的数据吗？

#### 1.6.1 任务一：利用Pandas对示例数据进行排序，要求升序


```python
# 具体请看《利用Python进行数据分析》第五章 排序和排名 部分

#自己构建一个都为数字的DataFrame数据
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), 
                     index=['2', '1'], 
                     columns=['d', 'a', 'b', 'c'])
frame
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



【代码解析】

pd.DataFrame() ：创建一个DataFrame对象 

np.arange(8).reshape((2, 4)) : 生成一个二维数组（2*4）,第一列：0，1，2，3 第二列：4，5，6，7

index=['2, 1] ：DataFrame 对象的索引列

columns=['d', 'a', 'b', 'c'] ：DataFrame 对象的索引行

【问题】：大多数时候我们都是想根据列的值来排序,所以将你构建的DataFrame中的数据根据某一列，升序排列


```python
frame.sort_values(by='c', ascending=True)
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#让行索引升序排序
frame.sort_index()
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
#让列索引升序排序
frame.sort_index(axis=1)
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
#让列索引降序排序
frame.sort_index(axis=1, ascending=False)
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
      <th>d</th>
      <th>c</th>
      <th>b</th>
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
#让任选两列数据同时降序排序
frame.sort_values(by=['a', 'c'], ascending=False)
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
      <th>d</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


#### 1.6.2 任务二：对泰坦尼克号数据（trian.csv）按票价和年龄两列进行综合排序（降序排列），从这个数据中你可以分析出什么？


```python

text.sort_values(by=['票价', '年龄'], ascending=False).head(10)
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>679</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
    </tr>
    <tr>
      <th>258</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>737</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
    </tr>
    <tr>
      <th>438</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>341</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>88</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>742</th>
      <td>743</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Susan Parker "Suzette"</td>
      <td>female</td>
      <td>21.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>311</th>
      <td>312</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>299</th>
      <td>300</td>
      <td>1</td>
      <td>1</td>
      <td>Baxter, Mrs. James (Helene DeLaudeniere Chaput)</td>
      <td>female</td>
      <td>50.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



【思考】排序后，如果我们仅仅关注年龄和票价两列。根据常识我知道发现票价越高的应该客舱越好，所以我们会明显看出，票价前20的乘客中存活的有14人，这是相当高的一个比例，那么我们后面是不是可以进一步分析一下票价和存活之间的关系，年龄和存活之间的关系呢？当你开始发现数据之间的关系了，数据分析就开始了。


```python
text.sort_values(by=['票价', '年龄']).head(20)
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>姓名</th>
      <th>性别</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>船票信息</th>
      <th>票价</th>
      <th>客舱</th>
      <th>登船港口</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>302</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>271</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>179</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>822</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>806</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0000</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>263</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0000</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>277</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>413</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>466</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>481</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>633</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>674</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>732</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>815</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>378</th>
      <td>379</td>
      <td>0</td>
      <td>3</td>
      <td>Betros, Mr. Tannous</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>2648</td>
      <td>4.0125</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>872</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>326</th>
      <td>327</td>
      <td>0</td>
      <td>3</td>
      <td>Nysveen, Mr. Johan Hansen</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>345364</td>
      <td>6.2375</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>843</th>
      <td>844</td>
      <td>0</td>
      <td>3</td>
      <td>Lemberopolous, Mr. Peter L</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>2683</td>
      <td>6.4375</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>818</th>
      <td>819</td>
      <td>0</td>
      <td>3</td>
      <td>Holm, Mr. John Fredrik Alexander</td>
      <td>male</td>
      <td>43.0</td>
      <td>0</td>
      <td>0</td>
      <td>C 7075</td>
      <td>6.4500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


#### 1.6.3 任务三：利用Pandas进行算术计算，计算两个DataFrame数据相加结果


```python
frame1_a = pd.DataFrame(np.arange(9.).reshape(3, 3),
                     columns=['a', 'b', 'c'],
                     index=['one', 'two', 'three'])
frame1_b = pd.DataFrame(np.arange(12.).reshape(4, 3),
                     columns=['a', 'e', 'c'],
                     index=['first', 'one', 'two', 'second'])
frame1_a
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>two</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>6.0</td>
      <td>7.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#两个DataFrame相加后，会返回一个新的DataFrame，对应的行和列的值会相加，没有对应的会变成空值NaN。
frame1_a + frame1_b
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>one</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>second</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>three</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>two</th>
      <td>9.0</td>
      <td>NaN</td>
      <td>13.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.6.4 任务四：通过泰坦尼克号数据如何计算出在船上最大的家族有多少人？


```python

max(text['兄弟姐妹个数'] + text['父母子女个数'])
```


    10



```python

max(text['年龄'] + text['票价'])
```


    548.3292


#### 1.6.5 任务五：学会使用Pandas describe()函数查看数据基本统计信息


```python

frame2 = pd.DataFrame([[1.4, np.nan], 
                       [7.1, -4.5],
                       [np.nan, np.nan], 
                       [0.75, -1.3]
                      ], index=['a', 'b', 'c', 'd'], columns=['one', 'two'])
frame2

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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.40</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>7.10</td>
      <td>-4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>0.75</td>
      <td>-1.3</td>
    </tr>
  </tbody>
</table>
</div>



调用 describe 函数，观察frame2的数据基本信息


```python

frame2.describe()
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
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.083333</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.493685</td>
      <td>2.262742</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.750000</td>
      <td>-4.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.075000</td>
      <td>-3.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.400000</td>
      <td>-2.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.250000</td>
      <td>-2.100000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.100000</td>
      <td>-1.300000</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.6.6 任务六：分别看看泰坦尼克号数据集中 票价、父母子女 这列数据的基本统计数据，你能发现什么？


```python

text['票价'].describe()
```


    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    25%        7.910400
    50%       14.454200
    75%       31.000000
    max      512.329200
    Name: 票价, dtype: float64



```python
text["票价"].describe(percentiles=[.75, .8])
```


    count    891.000000
    mean      32.204208
    std       49.693429
    min        0.000000
    50%       14.454200
    75%       31.000000
    80%       39.687500
    max      512.329200
    Name: 票价, dtype: float64



```python
text.describe(include=['O'])
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
      <th>乘客ID</th>
      <th>是否幸存</th>
      <th>仓位等级</th>
      <th>年龄</th>
      <th>兄弟姐妹个数</th>
      <th>父母子女个数</th>
      <th>票价</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>


#分析
- parameters：percentiles,include(计算离散型变量的统计特征),exclude

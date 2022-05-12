@[TOC](Notebook基础实践
)
# 实验内容

安装Jupyter Notebook和相关的Python环境，建议采用
Anaconda的安装方式。
• 按照教程完成实验过程，主要包括几个方面：
• 掌握Notebook工具的基本原理
• 学习Python基本语法，完成选择排序程序
• 完成Python数据分析的例子
• 将上述完成的Jupyter Notebook在Github上进行共享


## 实验步骤
### 1、创建一个新的Notebook
Anaconda -> Jupyter -> launch

[![O0GKET.jpg](https://img-blog.csdnimg.cn/img_convert/e6d0f8f8f0445f9eeb55db3a2fd7c0af.png)](https://imgtu.com/i/O0GKET)

new->Python 3 ipykernel创建

[![O0GmD0.jpg](https://img-blog.csdnimg.cn/img_convert/742cbe87331373d5cdc9edaaa4973355.png)](https://imgtu.com/i/O0GmD0)

### 2、试执行代码，查看效果
```python
print('Hello World!')
```

    Hello World!
    

```python
import time
time.sleep(3)
```
如图cell左侧的标签从In [ ] 变成了 In [*]，代表程序正在执行

[![O0J2l9.jpg](https://img-blog.csdnimg.cn/img_convert/f67e1680794fd902023ac36047883e84.png)](https://imgtu.com/i/O0J2l9)


### 3、Kernel
Kernel中运行的状态在整个文档中是延续的，可以跨越所有的cell
```python
import numpy as np
def square(x):
    return x * x
```


```python
x = np.random.randint(1, 10)
y = square(x)
print('%d squared is %d' % (x, y))
```

    5 squared is 25
    

### 4、简单的Python例子
完成基于python的选择排序算法：
1、定义selection_sort函数执行选择排序功能。
2、定义test函数进行测试，执行数据输入，并调用selection_sort函数进行排序，最后输出结果。

```python
# 选择排序法
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[min_index], arr[i] = arr[i], arr[min_index]
```


```python
def test():
    array = [26, 11, 99 , 33, 69, 77, 55, 56, 67]
    print("原列表：")
    print(array)
    selection_sort(array)
    print("排序后的列表：")
    print(array)
```


```python
test()
```

    原列表：
    [26, 11, 99, 33, 69, 77, 55, 56, 67]
    排序后的列表：
    [11, 26, 33, 55, 56, 67, 69, 77, 99]
    

### 5、数据分析的例子

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('fortune500.csv')
```

检查数据集
```python
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
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.tail()
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
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25495</th>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <th>25496</th>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <th>25497</th>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <th>25498</th>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <th>25499</th>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>
对数据属性列进行重命名

```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```
检查数据条目是否加载完整
```python
len(df)
```




    25500


检查属性列的类型

```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object
```python
non_numberic_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numberic_profits].head()
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
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>228</th>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>290</th>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>294</th>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>296</th>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <th>352</th>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>

统计存在多少条记录
```python
len(df.profit[non_numberic_profits])
```




    369

使用直方图显示一下按照年份的分布情况
```python
bin_sizes, _, _ = plt.hist(df.year[non_numberic_profits], bins=range(1955, 2006))
```


[![O0GMUU.png](https://img-blog.csdnimg.cn/img_convert/02a6571282abdf4b09111fe2d47db135.png)](https://imgtu.com/i/O0GMUU)

删除记录
```python
df = df.loc[~non_numberic_profits]
df.profit = df.profit.apply(pd.to_numeric)
```

检查数据记录的条目数
```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object



### 6、使用matplotlib进行绘图
```python
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x=0, y=0)
```

开始绘图
```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


[![O0GQ5F.png](https://img-blog.csdnimg.cn/img_convert/6d9760422f7322819455be3db7fd4106.png)](https://imgtu.com/i/O0GQ5F)

```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')
```


    
[![O0G1C4.png](https://img-blog.csdnimg.cn/img_convert/ef57a9e925113ffa105e7f31ad3937fb.png)](https://imgtu.com/i/O0G1C4)
    


对数据结果进行标准差处理
```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha=0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols=2)
title = 'Increase in mean and std Fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()
```


[![O0G829.png](https://img-blog.csdnimg.cn/img_convert/bb3a5fc229dab4109263d6b5bd5ce30d.png)](https://imgtu.com/i/O0G829)


### 7、分享Notebooks
Cell > All Output > Clear

[![O0Geuq.jpg](https://img-blog.csdnimg.cn/img_convert/364747e80bf65a3642cb79473e9f93c9.png)](https://imgtu.com/i/O0Geuq)

Kernel > Restart & Run All

[![O0GVvn.jpg](https://img-blog.csdnimg.cn/img_convert/491ed6e6a55d4259a52b2143f9530535.png)](https://imgtu.com/i/O0GVvn)

导出Notebooks

[![O0GnbV.jpg](https://img-blog.csdnimg.cn/img_convert/3caa950946563fce806eddc094a2e8a1.png)](https://imgtu.com/i/O0GnbV)


### 8、Jupyter Notebook扩展工具
Anaconda Navigator中启动命令行终端，输入以下安装代码：

```kotlin
pip install jupyter_contrib_nbextensions

jupyter contrib nbextension install --user

pip install jupyter_nbextensions_configurator

jupyter nbextensions_configurator enable --user
```

点击Nbextensions标签，勾选Hinterland即可

[![O0G38J.jpg](https://img-blog.csdnimg.cn/img_convert/969cb2cb2b593cbd7489a875f2e875b4.png)](https://imgtu.com/i/O0G38J)



作者：员力
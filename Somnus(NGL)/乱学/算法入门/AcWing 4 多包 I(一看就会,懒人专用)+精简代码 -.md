# AcWing 4. 多包 I(一看就会,懒人专用)+精简代码 - AcWing（强拆法--粗暴好用）

### 题目描述

有 N种物品和一个容量是 V的背包。

第 i种物品最多有 si件，每件体积是 vi，价值是 wi。

求解将哪些物品装入背包，可使物品体积总和不超过背包容量，且价值总和最大。

输出最大价值。

输入格式

第一行两个整数，N，V，用空格隔开，分别表示物品种数和背包容积。

接下来有 N 行，每行三个整数 vi,wi,si，用空格隔开，分别表示第 i 种物品的体积、价值和数量。

输出格式

输出一个整数，表示最大价值。

数据范围

0<N,V≤100

0<vi,wi,si≤100

### 样例

```
输入样例
4 5
1 2 3
2 4 1
3 4 3
4 5 2
输出样例：
10
```

---

### 分析

当 si=1 时，相当于01背包中的一件物品

当 si>1 时，相当于01背包中的多个一件物品

故我们可以死拆(把多重背包拆成01背包)

## ~~别忘了Star~~

### C++ 代码

```cpp
#include <bits/stdc++.h>
using namespace std;
int a[10005],b[10005],t=0,n,m,dp[10005]={ },w,v,s;
int main(){
    cin>>n>>m;
    while(n--){        
		    cin>>v>>w>>s;    
		    while(s--){       
		       a[++t]=v;
		       b[t]=w;        }//死拆，把多重背包拆成01背包   
		 }    
		for(int i=1;i<=t;i++)
		    for(int j=m;j>=a[i];j--)
		        dp[j]=max(dp[j-a[i]]+b[i],dp[j]);//直接套01背包的板子
		cout<<dp[m]<<endl;
		return 0;
		}
```

## ~~优化，稍微短了一点~~(把for改成while)（

### 简化代码

```cpp
#include <bits/stdc++.h>
using namespace std;
int dp[1005],n,t,v,w,s;
main(){    
		cin>>n>>t;    
		while(n--){    
		cin>>w>>v>>s;    
		while(s--){    
				for(int j=t;j>=w;j--){
				dp[j]=max(dp[j],dp[j-w]+v);
		    }
    }    
    cout<<dp[t];
}
```

其实死拆还有后面的01背包不用分开,当成s个物品做就可以了。

```cpp
#include <bits/stdc++.h>
using namespace std;
int a[10005],b[10005],t=0,n,m,dp[10005]={ },w,v,s;
int main(){    
  cin>>n>>m;    
  while(n--){        
  cin>>v>>w>>s;       
    while(s--){           
     for(int j=m;j>=v;j--)         
     dp[j]=max(dp[j-v]+w,dp[j]);//直接套01背包的板子 
                }
          } 
   cout<<dp[m]<<endl;    
   return 0;
}
```

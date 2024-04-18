
## 相比于01背包的改变：每个物品可以选取无数次

[[AcWing 4 多包 I(一看就会,懒人专用)+精简代码 -]]

• 闫氏DP分析法：[b站讲解](https://www.bilibili.com/video/BV1X741127ZM)


```cpp
#include <bits/stdc++.h>


using namespace std;

int f[1200][1200];
int si[1200];
int main(){
	int N,V;
	cin>>N>>V;
	int v[N+4],w[N+4];
	for(int i=1;i<=N;i++){
		cin>>v[i]>>w[i];
	}
	for(int i=1;i<=N;i++){
		for(int j=0;j<=V;j++){
		    if(j>=v[i])//能放下 
			f[i][j]=max(f[i-1][j],f[i][j-v[i]]+w[i]);//要么不放，要么放一个-->用上一个递推 
			else
			f[i][j]=f[i-1][j];//放不下
		}
	}	
	cout<<f[N][V]<<endl;
	
//一维数组可行吗？
//与01背包完全不同，我们使用递推的方式不断更新si的内容
//正好需要si[j-v[i]]的内容是物品i的已被更新的状态 
 
	for(int i=1;i<=N;i++){
		for(int j=v[i];j<=V;j++){
			si[j]=max(si[j],si[j-v[i]]+w[i]);
		}
	}
	cout<<si[V]<<endl;
	
	return 0;
}
```


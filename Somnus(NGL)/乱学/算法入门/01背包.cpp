#include <bits/stdc++.h>
using namespace std;

int N,V;

int f[1200][1200];  //f[I][J]表示对于I个物品、J个空间下的最优价值

int si[1200]; //一维数组简化 

//main 外声明，自动初始化为0； 

int main(){
	cin>>N>>V;  //输入N个物品、背包空间为V 
	int v[1200],w[1200];	//v-->每个物体体积   w-->每个物体的价值 
	for(int i=1;i<=N;i++){
		scanf("%d %d",&v[i],&w[i]);//意义-->从第一个物品开始记录每个物品的V、W，序号与i相同 
	}
	for(int i=0;i<N;i++){
	//	cout<<v[i]<<w[i]<<endl;
	}
	for(int i=1;i<=N;i++){
		for(int j=1;j<=V;j++){
			f[i][j]=0;
		}
	}
	//核心是递归的思想，自前向后保证计算时有值 而非 0 
	for(int i=1;i<=N;i++){//00已经初始化为0，所以可以掠过 
		for(int j=1;j<=V;j++){
			if(j<v[i]){   //首先考虑能否放下第I个物品 
				f[i][j]=f[i-1][j];//若不能放下，即相当于对于i-1个物品，j个空间下的情况 
			}else{
				f[i][j]=max(f[i-1][j],f[i-1][j-v[i]]+w[i]);//若能放下，则需比较不放他和放下它的情况 哪个更值 
			}
		//	cout<<"f[i][j]="<<f[i][j]<<endl;
		}
	}
	cout<<f[N][V]<<endl;	
	//以下提供一维数组的解决方案:通过后向前遍历让i-1的状态决定i的状态 
	
	for(int i=1;i<=N;i++){
		for(int j=V;j>=v[i];j--){
			si[j]=max(si[j],si[j-v[i]]+w[i]);
		}
	}
	cout<<si[V]<<endl
    return 0;
}

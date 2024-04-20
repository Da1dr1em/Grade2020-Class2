#include <bits/stdc++.h>
using namespace std;
int a[10005],b[10005],t=0,n,m,multpack[10005]={   },w,v,s;
int main(){
    cin>>n>>m;   // 输入物品个数和背包体积
    while(n--){
        cin>>v>>w>>s;
        while(s--){        //展开为01背包
            t++;
            a[t]=v;
            b[t]=w;
        }
    }
    for(int i=1;i<=t;i++){//01背包的处理方法
        for(int j=m;j>=a[i];j--){
            multpack[j]=max(multpack[j],multpack[j-a[i]]+b[i]);

        }
    }
    cout<<multpack[m]<<endl;

    return 0;
}

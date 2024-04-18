本关任务：设圆半径r，圆柱高h ， 求圆周长C1，半径为r的圆球表面积Sb,圆半径r，圆柱高为h的圆柱体积Vb。 用scanf输入数据，输出计算结果，输出时取小数点后两位数字。请编程序。 PI＝3.14
   #include<stdio.h>
	int main( )
	{  
	double r,h,C1,Sb,Vb,pi;
    pi=3.14;
    scanf("%lf,%lf",&r,&h);
    C1=pi*2*r;
    Sb=4*pi*r*r;
    Vb=pi*r*r*h;
    printf("C1=%.2f\nSb=%.2f\nVb=%.2f\n",C1,Sb,Vb);
       return 0;
	}

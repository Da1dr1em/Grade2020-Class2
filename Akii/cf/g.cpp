#include <bits/stdc++.h>
using namespace std;
using ll = long long;
char a[20][1009];
char b[20][1009];
int c[20], d[20];

int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int i;
	cin >> i;
	for (int j = 1; j <= i; j++) {
		cin >> a[j];
		cin >> b[j];
		for (int z = 1; z <= j; z++) {
			if (a[z] == a[j]) {
				c[z]++;
				break;
			}
		}
		for (int z = 1; z <= j; z++) {
			if (b[z] == b[j]) {
				d[z]++;
				break;
			}
		}
	}


	return 0;
}
#include <bits/stdc++.h>
using namespace std;
using ll = long long;


int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int t;
	cin >> t;
	while (t > 0) {
		t--;
		int n;
		cin >> n;
		int a[109];
		for (int i = 1; i <= 105; i++)
			a[i] = 0;
		for (int i = 1; i <= n; i++) {
			int j;
			cin >> j;
			a[j]++;
		}
		int m = 0;
		for (int i = 1; i <= 105; i++) {
			while (a[i] >= 3) {
				m++;
				a[i] -= 3;
			}
		}
		cout << m << '\n';
	}

	return 0;
}

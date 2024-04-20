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
		int a[n];
		for (int i = 1; i <= n; i++)
			a[i] = 0;
		for (int i = 1; i <= n; i++) {
			int m;
			cin >> m;
			a[m]++;
		}
		int x = 0, y = 0, z = 0;
		for (int i = 1; i <= n; i++) {
			if (a[i] == 0)
				x++;
			else if (a[i] == 2)
				z++;
		}
		if (x < z) {
			cout << n - x;
		}
		if (x >= z) {
			cout << (z);
		}
		cout << '\n';
	}

	return 0;
}
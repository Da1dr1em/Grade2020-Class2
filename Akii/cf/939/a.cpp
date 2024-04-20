#include <bits/stdc++.h>
using namespace std;
using ll = long long;


int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int t;
	cin >> t;
	while (t > 0) {
		t--;
		int k, q;
		cin >> k >> q;
		int a[k];
		for (int i = 1; i <= k; i++ ) {
			cin >> a[i];
		}
		for (int i = 1; i <= q; i++) {
			int n;
			cin >> n;
			if (n < a[1])
				cout << n;
			else
				cout << a[1] - 1;
			cout << ' ';
		}
		cout << '\n';
	}

	return 0;
}
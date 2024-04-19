#include <bits/stdc++.h>
using namespace std;
using ll = long long;


int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int t;
	cin >> t;
	while (t > 0) {
		t--;
		int n, a, b;
		cin >> n >> a >> b;
		if ((2 * a) <= b)
			cout << n *a;
		else {
			int c;
			c = n % 2;
			if (c == 0)
				cout << n / 2 * b;
			else
				cout << n / 2 * b + a;

		}
		cout << '\n';
	}
	return 0;
}

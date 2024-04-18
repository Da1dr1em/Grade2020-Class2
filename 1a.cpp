#include <bits/stdc++.h>
using namespace std;
using ll = long long;


int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int i;
	cin >> i;
	for (int j = 1; j <= i; j++) {


		int a, b, c;
		cin >> a >> b >> c;
		if ((a >= b) || (b == c))
			cout << "NONE";
		else {
			if (b > c)
				cout << "PEAK";
			else
				cout << "STAIR";
		}
		cout << '\n';

	}
	return 0;
}
#include <bits/stdc++.h>
using namespace std;
using ll = long long;

void quick_sort(int q[], int l, int r) {
	if (l >= r)
		return;

	int i = l - 1, j = r + 1, x = q[l + r >> 1];
	while (i < j) {
		do
			i ++ ;
		while (q[i] < x);
		do
			j -- ;
		while (q[j] > x);
		if (i < j)
			swap(q[i], q[j]);
	}
	quick_sort(q, l, j), quick_sort(q, j + 1, r);
}

int main() {
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	int t;
	cin >> t;
	while (t > 0) {
		t--;
		int n, c, d;
		cin >> n >> c >> d;
		int a[n * n];
		int q = 0;
		for (int i = 1; i <= (n * n); i++)
			cin >> a[i];
		for (int i = 1; i <= (n * n); i++)
			q = q + a[i];
		quick_sort(a, 1, n * n);
		int x = n * (n - 1) / 2;
		q = q - x * c * n;
		q = q - x * d * n;
		if ((q / a[1]) == (n * n))
			cout << "yes";
		else
			cout << "no";
		cout << "\n";


	}
	return 0;
}
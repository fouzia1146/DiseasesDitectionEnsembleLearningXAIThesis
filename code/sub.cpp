#include <iostream>
#include <cmath>
using namespace std;
#define endl "\n"
#define ll long long int
#define yes cout<<"YES"<<endl
#define no cout<<"NO"<<endl
 
void solve()
{
	ll a,s=0;
	cin>>a;
	s=a%10+a/10;
	cout<<s<<endl;
}
int main()
{
	int test=1;
	cin>>test;
	while(test--)
	{
		solve();
	}
}
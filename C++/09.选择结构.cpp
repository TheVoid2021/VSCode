#include <iostream>
using namespace std;

int main()
{

  int A = 0;
  int B = 0;
  int C = 0;

  cout << "������С��A�����أ�" << endl;
  cin >> A;
  cout << "������С��B�����أ�" << endl;
  cin >> B;
  cout << "������С��C�����أ�" << endl;
  cin >> C;

  int MAXmight = 0;
  if (A > B)
  {
    MAXmight = A;
    if (MAXmight > C)
    {
      cout << "���ص�С����:AС��" << endl;
    }
    else
    {
      MAXmight = C;
      cout << "���ص�С����:CС��" << endl;
    }
  }
  else
  {
    MAXmight = B;
    if (MAXmight > C)
    {
      cout << "���ص�С����:BС��" << endl;
    }
    else
    {
      MAXmight = C;
      cout << "���ص�С����:CС��" << endl;
    }
  }
  system("pause");
  return 0;
}
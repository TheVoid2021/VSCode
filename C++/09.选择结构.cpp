#include <iostream>
using namespace std;

int main()
{

  int A = 0;
  int B = 0;
  int C = 0;

  cout << "请输入小猪A的体重：" << endl;
  cin >> A;
  cout << "请输入小猪B的体重：" << endl;
  cin >> B;
  cout << "请输入小猪C的体重：" << endl;
  cin >> C;

  int MAXmight = 0;
  if (A > B)
  {
    MAXmight = A;
    if (MAXmight > C)
    {
      cout << "最重的小猪是:A小猪" << endl;
    }
    else
    {
      MAXmight = C;
      cout << "最重的小猪是:C小猪" << endl;
    }
  }
  else
  {
    MAXmight = B;
    if (MAXmight > C)
    {
      cout << "最重的小猪是:B小猪" << endl;
    }
    else
    {
      MAXmight = C;
      cout << "最重的小猪是:C小猪" << endl;
    }
  }
  system("pause");
  return 0;
}
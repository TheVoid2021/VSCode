/*
* 数组元素逆置
案例描述：请声明一个5个元素的数组，并且将元素逆置.
(如原数组元素为：1,3,2,5,4;逆置后输出结果为:4,5,2,3,1);
*/
#include <iostream>
using namespace std;

int main()
{

  int arr[5];
  cout << "请输入五个数: " << endl;
  for (int i = 0; i < 5; i++)
  {
    cin >> arr[i];
  }
  cout << "逆置后的结果为：" << endl;
  for (int j = 4; j < 5 && j > -1; j--)
  {
    cout << arr[j] << " ";
  }

  system("pause");
  return 0;
}
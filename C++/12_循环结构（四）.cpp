/*
* 练习案例：敲桌子

案例描述：从1开始数到数字100， 如果数字个位含有7，或者数字十位含有7，或者该数字是7的倍数，我们打印敲桌子，其余数字直接打印输出。
*/
#include <iostream>
using namespace std;

int main()
{
  int a = 0; // 各位
  int b = 0; // 十位
  int c = 0; // 检查是否7的倍数

  for (int i = 1; i <= 100; i++)
  {
    a = i % 10;      // 获得i的各位
    b = i / 10 % 10; // 获得i的十位
    c = i % 7;       // 获得i除以7的余数
    if (a == 7 || b == 7 || c == 0)
    {
      cout << "敲桌子" << i << endl;
    }
    else
    {
      cout << i << endl;
    }
  }
  system("pause");
  return 0;
}
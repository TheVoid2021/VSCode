#include <iostream>
using namespace std;

int main()
{

  // 加减乘除
  int a1 = 10;
  int b1 = 3;
  /*
    cout << a1 + b1 << endl;
    cout << a1 - b1 << endl;
    cout << a1 * b1 << endl;
    cout << a1 / b1 << endl;*/
  // 两个整数相除 结果依然是整数，将小数部分去除 除数不能为0

  // 1、前置递增
  int a = 10;
  ++a;
  cout << "a=" << a << endl;

  // 2、后置递增
  int b = 10;
  b++;
  cout << "b=" << b << endl;

  // 3、前置和后置的区别
  // 前置递增 先让变量+1 然后进行表达式运算
  int a2 = 10;
  int b2 = ++a2 * 10;
  cout << b2 << endl;
  cout << a2 << endl;
  // 后置递增 先进行表达式运算，后让变量+1
  int a3 = 10;
  int b3 = a3++ * 10;

  cout << b3 << endl;
  cout << a3 << endl;

  // 逻辑或
  int a4 = 10;
  int a5 = 0;
  cout << (a4 || a5) << endl;

  system("pause");

  return 0;
}
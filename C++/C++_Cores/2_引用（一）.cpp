#include <iostream>
using namespace std;

/*
!引用
*作用： 给变量起别名——&

*语法： `数据类型 &别名 = 原名` */

int main()
{

  int a = 10;
  int &b = a; // !一旦初始化后，就不可以更改
  int c = 20;

  // int &c; // !错误，引用必须初始化

  b = c; // !这是赋值操作，不是更改引用，引用本身没有开辟内存空间，它只是原变量别名

  // 访问同一块内存空间，修改一个，另一个也会改变
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "c = " << c << endl;

  b = 100;

  cout << "a = " << a << endl;
  cout << "b = " << b << endl;

  system("pause");

  return 0;
}
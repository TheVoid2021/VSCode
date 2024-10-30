#include <iostream>
using namespace std;

int main()
{

  // 1、指针的定义
  int a = 10; // 定义整型变量a

  // 指针定义语法： 数据类型 * 变量名 ;
  int *p; // !“*”  ：指针变量标识符

  // 指针变量赋值
  p = &a;             // !指针指向变量a的地址  &：取地址符
  cout << &a << endl; // 打印数据a的地址
  cout << p << endl;  // 打印指针变量p

  // 2、指针的使用
  // 通过*操作指针变量指向的内存
  cout << "*p = " << *p << endl; // !指针前面加*：解引用操作，获取指针指向的内存中的数据

  system("pause");

  return 0;
}
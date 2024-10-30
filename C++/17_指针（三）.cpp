#include <iostream>
using namespace std;

int main()
{

  // 指针变量p指向内存地址编号为0的空间
  int *p = NULL;

  // 访问空指针报错
  // !内存编号0 ~255为系统占用内存，不允许用户访问
  cout << *p << endl;

  // 指针变量p指向内存地址编号为0x1100的空间
  int *q = (int *)0x1100;

  // 访问野指针报错
  cout << *q << endl;

  int a = 10;
  int b = 10;

  // !技巧：看const右侧紧跟着的是指针还是常量, 是指针就是常量指针，是常量就是指针常量
  // !const修饰的是指针，指针指向可以改，指针指向的值不可以更改
  const int *p1 = &a; // 常量指针
  p1 = &b;            // 正确
  //*p1 = 100;  报错

  // !const修饰的是常量，指针指向不可以改，指针指向的值可以更改
  int *const p2 = &a; // 指针常量
  // p2 = &b; //错误
  *p2 = 100; // 正确

  // !const既修饰指针又修饰常量，指针指向和指针指向的值都不可以改
  const int *const p3 = &a; // 常量指针常量
  // p3 = &b; //错误
  //*p3 = 100; //错误

  system("pause");

  return 0;
}
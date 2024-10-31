#include <iostream>
using namespace std;

/*
!引用做函数参数

*作用：函数传参时，可以利用引用的技术让形参修饰实参

*优点：可以简化指针修改实参 */

/*
!引用做函数返回值

*作用：引用是可以作为函数的返回值存在的

*注意：不要返回局部变量引用

*用法：函数调用作为左值 */

// !1. 值传递
void mySwap01(int a, int b)
{
  int temp = a;
  a = b;
  b = temp;
}

// !2. 地址传递
void mySwap02(int *a, int *b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}

// !3. 引用传递
void mySwap03(int &a, int &b)
{
  int temp = a;
  a = b;
  b = temp;
}

//! 返回局部变量引用
// int &test01()
// {
//   int c = 10; // 局部变量
//   return c;
// }

//! 返回静态变量引用
int &test02()
{
  static int a = 20;
  return a;
}

//! 引用的本质在c++内部实现是一个指针常量.
//  发现是引用，转换为 int* const ref = &a;
void func(int &ref)
{
  ref = 100; // ref是引用，转换为*ref = 100
}

//! 常量引用主要用来修饰形参，防止误操作
//! 在函数形参列表中，可以加const修饰形参，防止形参改变实参
// 引用使用的场景，通常用来修饰形参
void showValue(const int &v)
{
  // v += 10;
  cout << v << endl;
}

int main()
{

  int a = 10;
  int b = 20;

  // todo值传递，形参不会修饰实参
  mySwap01(a, b);
  cout << "a:" << a << " b:" << b << endl;

  // todo地址传递，形参会修饰实参
  mySwap02(&a, &b);
  cout << "a:" << a << " b:" << b << endl;

  // todo引用传递，形参会修饰实参
  mySwap03(a, b);
  cout << "a:" << a << " b:" << b << endl;

  // todo不能返回局部变量的引用
  // int &ref = test01();
  // cout << "ref = " << ref << endl; // 第一次结果正确，是因为编译器做了保留
  // cout << "ref = " << ref << endl; // 第二次结果错误，因为局部变量a已经释放

  // todo如果函数做左值，那么必须返回引用
  int &ref2 = test02();
  cout << "ref2 = " << ref2 << endl; // 0
  cout << "ref2 = " << ref2 << endl; //  test02() = 100;

  test02() = 1000;

  cout << "ref2 = " << ref2 << endl;
  cout << "ref2 = " << ref2 << endl;

  int c = 10;

  // 自动转换为 int* const ref = &a; 指针常量是指针指向不可改，也说明为什么引用不可更改
  int &ref = c;
  ref = 20; // 内部发现ref是引用，自动帮我们转换为: *ref = 20;

  cout << "c:" << c << endl;
  cout << "ref:" << ref << endl;

  func(c);

  // int& ref = 10;  引用本身需要一个合法的内存空间，因此这行错误
  // 加入const就可以了，编译器优化代码，int temp = 10; const int& ref = temp;
  const int &ref1 = 10;

  // ref = 100;  //加入const后不可以修改变量
  cout << ref1 << endl;

  // 函数中利用常量引用防止误操作修改实参
  int d = 10;
  showValue(d);

  system("pause");

  return 0;
}

#include <iostream>
using namespace std;

/*
! 模板就是建立 *通用的模具* ，大大提高 复用性
? 模板的特点：
  * 模板不可以直接使用，它只是一个框架
  * 模板的通用并不是万能的
todo C++另一种编程思想称为 =泛型编程= ，主要利用的技术就是模板
todo C++提供两种模板机制: *函数模板* 和 *类模板*

? 函数模板作用：
  * 建立一个通用函数，其函数返回值类型和形参类型可以不具体制定，用一个 *虚拟的类型* 来代表。
todo 语法：
    * template<typename T>
    * 函数声明或定义
todo 解释：
    * template  ---  声明创建模板
    * typename  --- 表面其后面的符号是一种数据类型，可以用class代替
    * T    ---   通用的数据类型，名称可以替换，通常为大写字母
 */

// 交换整型函数
void swapInt(int &a, int &b)
{
  int temp = a;
  a = b;
  b = temp;
}

// 交换浮点型函数
void swapDouble(double &a, double &b)
{
  double temp = a;
  a = b;
  b = temp;
}

// ! 利用模板提供通用的交换函数
template <typename T> // todo 声明一个模板，告诉编译器后面代码中紧跟着的代码中数据类型参数T是通用的
void mySwap(T &a, T &b)
{
  T temp = a;
  a = b;
  b = temp;
}

void test01()
{
  int a = 10;
  int b = 20;

  // swapInt(a, b);

  // todo 利用模板实现交换 (两种方式)
  // * 1、自动类型推导
  mySwap(a, b);

  // * 2、显示指定类型
  mySwap<int>(a, b);

  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
}

// todo 利用模板提供通用的交换函数
template <class T> // todo 表面其后面的符号是一种数据类型，typename可以用class代替
void mySwap1(T &a, T &b)
{
  T temp = a;
  a = b;
  b = temp;
}

// ! 1、自动类型推导，必须推导出一致的数据类型T,才可以使用
void test02()
{
  int a = 10;
  int b = 20;
  // char c = 'c';

  mySwap1(a, b); // 正确，可以推导出一致的T
                 // mySwap(a, c); // 错误，推导不出一致的T类型
}

// ! 2、模板必须要确定出T的数据类型，才可以使用
template <class T>
void func()
{
  cout << "func 调用" << endl;
}

void test03()
{
  // func(); //错误，模板不能独立使用，必须确定出T的类型
  func<int>(); // 利用显示指定类型的方式，给T一个类型，才可以使用该模板
}

int main()
{

  test01();

  test02();

  test03();

  system("pause");

  return 0;
}
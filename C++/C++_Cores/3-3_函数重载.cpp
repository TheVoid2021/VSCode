#include <iostream>
using namespace std;

/*
!函数重载
*作用：函数名可以相同，提高复用性
*函数重载满足条件：
todo 同一个作用域下
todo 函数名称相同
todo 函数参数 **类型不同**  或者 **个数不同** 或者 **顺序不同**

*注意:  函数的返回值不可以作为函数重载的条件 */

// void func()
// {
//   cout << "func调用" << endl;
// }

// void func(int a)
// {
//   cout << "func(int a)调用" << endl;
// }

// void func(double a)
// {
//   cout << "func(double a)调用" << endl;
// }

// void func(int a, double b)
// {
//   cout << "func(int a, double b)调用" << endl;
// }

// void func(double b, int a)
// {
//   cout << "func(double b, int a)调用" << endl;
// }

// !函数的返回值不可以作为函数重载的条件
//  int func(int a){
//      cout << "func(int a)调用" << endl;
//      return 0;
// }

//! 函数重载注意事项
//* 引用作为重载条件
// void func1(int &a) // int &a = 10; 不合法，引用必须引用合法的内存空间
// {
//   cout << "func1(int &a)调用" << endl;
// }

// void func1(const int &a)
// {
//   cout << "func1(const int &a)调用" << endl;
// }
//* 函数重载碰到函数默认参数
// void func2(int a, int b = 10) // 函数重载碰到函数默认参数，可以调用上面的函数
// {
//   cout << "func2(int a)调用" << endl;
// }

// void func2(int a) // 也可以调用下面的函数
// {
//   cout << "func2(int a)调用" << endl;
// }

int main()
{
  // func();
  // func(10);
  // func(3.14);
  // func(10, 3.14);
  // func(3.14, 10);

  // int a = 10;
  // func1(a); //不加const引用调用
  // func1(10); // 加了const引用调用

  // func2(10); // 函数重载碰到函数默认参数，出现二义性，所以编译器会报错，尽量避免出现这样的情况
  system("pause");
  return 0;
}
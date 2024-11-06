#include <iostream>
using namespace std;

/*
!静态成员就是在成员变量和成员函数前加上关键字static，称为静态成员
?静态成员分为：
  todo静态成员变量
   *  所有对象共享同一份数据
   *  在编译阶段分配内存 全局区
   *  类内声明，类外初始化
  todo静态成员函数
   *  所有对象共享同一个函数
   *  静态成员函数只能访问静态成员变量
 */

class Person
{
public:
  // todo类内声明
  static int m_A; // !静态成员变量定义

  // 静态成员变量特点：
  // 1 在编译阶段分配内存
  // 2 类内声明，类外初始化
  // 3 所有对象共享同一份数据

  // 静态成员函数特点：
  // 1 程序共享一个函数
  // 2 静态成员函数只能访问静态成员变量

  static void func() // !静态成员函数定义
  {
    cout << "func调用" << endl;
    m_C = 100;
    // m_D = 100; //!错误，不可以访问非静态成员变量
  }

  static int m_C; // 静态成员变量
  int m_D;        // 非静态成员变量

private:
  // todo类内声明
  static int m_B; // ?静态成员变量也是有访问权限的

  // ?静态成员函数也是有访问权限的
  static void func2()
  {
    cout << "func2调用" << endl;
  }
};
int Person::m_A = 10; // todo类外初始化
int Person::m_B = 10; // todo类外初始化

int Person::m_C = 10; // todo类外初始化

void test01()
{
  // !静态成员变量两种访问方式

  // todo 1、通过对象
  Person p1;
  p1.m_A = 100;
  cout << "p1.m_A = " << p1.m_A << endl;

  Person p2;
  p2.m_A = 200;
  cout << "p1.m_A = " << p1.m_A << endl; // !共享同一份数据
  cout << "p2.m_A = " << p2.m_A << endl;

  // todo 2、通过类名
  cout << "m_A = " << Person::m_A << endl;

  // cout << "m_B = " << Person::m_B << endl; // ?私有权限访问不到

  // !静态成员变量两种访问方式

  // todo 1、通过对象
  Person p3;
  p3.func();

  // todo 2、通过类名
  Person::func();

  // Person::func2(); // ?私有权限访问不到
}

int main()
{

  test01();

  system("pause");

  return 0;
}
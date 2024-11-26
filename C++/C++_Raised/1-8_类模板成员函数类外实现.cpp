#include <iostream>
using namespace std;

/*
! 类模板成员函数类外实现
* 类模板中成员函数类外实现时，需要加上模板参数列表!

? 提问：构造函数和成员函数有什么区别？
  * 1.构造函数没有返回值类型，成员函数有返回值类型
  * 2.构造函数在对象创建的时候自动调用，成员函数在对象创建之后调用
  * 3.构造函数的函数名与类名相同，成员函数的函数名与类名不同
  * 4.构造函数只能有一个，成员函数可以有多个
  * 5.构造函数的参数列表可以没有，成员函数的参数列表可以有也可以没有
  * 6.构造函数的参数列表可以有默认值，成员函数的参数列表可以有也可以没有
 */

// 类模板中成员函数类外实现
template <class T1, class T2> // class也可以用typename代替
class Person
{
public:
  // todo 成员函数类内声明
  Person(T1 name, T2 age);
  void showPerson();

public:
  T1 m_Name;
  T2 m_Age;
};

// todo 构造函数 类外实现
template <class T1, class T2>
Person<T1, T2>::Person(T1 name, T2 age) // Person<T1, T2>表示这是一个类模板的构造函数
{
  this->m_Name = name;
  this->m_Age = age;
}

// todo 成员函数 类外实现
template <class T1, class T2>
void Person<T1, T2>::showPerson()
{
  cout << "姓名: " << this->m_Name << " 年龄:" << this->m_Age << endl;
}

void test01()
{
  Person<string, int> p("Tom", 20);
  p.showPerson();
}

int main()
{

  test01();

  system("pause");

  return 0;
}
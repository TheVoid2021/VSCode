#include <iostream>
using namespace std;

/*
! 类模板和函数模板语法相似，在声明模板template后面加类，此类称为类模板
! 类模板语法
? 类模板作用：
  * 建立一个通用类，类中的成员 数据类型可以不具体制定，用一个**虚拟的类型**来代表。
? 语法：
  * template<typename T>
  * 类
? 解释：
  * template  ---  声明创建模板
  * typename  --- 表面其后面的符号是一种数据类型，可以用class代替
  * T    ---   通用的数据类型，名称可以替换，通常为大写字母
 */

/*
! 类模板与函数模板区别
? 类模板与函数模板区别主要有两点：
  * 1. 类模板没有自动类型推导的使用方式，只能用显示指定类型
  * 2. 类模板在模板参数列表中可以有默认参数
 */

/*
! 类模板中成员函数创建时机
? 类模板中成员函数和普通类中成员函数创建时机是有区别的：
  * 普通类中的成员函数一开始就可以创建
  * 类模板中的成员函数在调用时才创建
 */

// ! 类模板
template <class NameType, class AgeType = int> // todo 类模板中可以定义默认参数
class Person
{
public:
  Person(NameType name, AgeType age)
  {
    this->mName = name;
    this->mAge = age;
  }
  void showPerson()
  {
    cout << "name: " << this->mName << " age: " << this->mAge << endl;
  }

public:
  NameType mName;
  AgeType mAge;
};

void test01()
{
  // 指定NameType 为string类型，AgeType 为 int类型
  Person<string, int> P1("孙悟空", 999);
  P1.showPerson();
}

// todo 1、类模板没有自动类型推导的使用方式
void test02()
{
  // Person p("孙悟空", 1000); // 错误 类模板使用时候，不可以用自动类型推导
  Person<string, int> p("孙悟空", 1000); // 必须使用显示指定类型的方式，使用类模板
  p.showPerson();
}

// todo 2、类模板在模板参数列表中可以有默认参数
void test03()
{
  Person<string> p("猪八戒", 999); // 类模板中的模板参数列表 可以指定默认参数
  p.showPerson();
}

class Person1
{
public:
  void showPerson1()
  {
    cout << "Person1 showPerson1" << endl;
  }
};

class Person2
{
public:
  void showPerson2()
  {
    cout << "Person2 showPerson2" << endl;
  }
};

template <class T>
class Myclass
{
public:
  T obj;

  // ! 类模板中的成员函数并不是一开始就创建的，而是在调用的时候才创建
  void func1()
  {
    obj.showPerson1();
  }

  void func2()
  {
    obj.showPerson2();
  }
};

void test01()
{
  Myclass<Person1> m1;
  m1.func1();
  m1.func2();
}

int main()
{

  test01();

  test02();

  test03();

  system("pause");

  return 0;
}
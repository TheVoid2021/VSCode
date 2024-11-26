#include <iostream>
using namespace std;

/*
! 类模板对象做函数参数
? 学习目标：
  * 类模板实例化出的对象，向函数传参的方式
? 一共有三种传入方式：
  * 1. 指定传入的类型   --- 直接显示对象的数据类型  (常用)
  * 2. 参数模板化           --- 将对象中的参数变为模板进行传递
  * 3. 整个类模板化       --- 将这个对象类型 模板化进行传递
 */

// 类模板
template <class NameType, class AgeType = int>
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

// ! 1、指定传入的类型
void printPerson1(Person<string, int> &p) // 用类模板实例化出的对象，向函数传参
{
  p.showPerson();
}
void test01()
{
  Person<string, int> p("孙悟空", 100);
  printPerson1(p);
}

// ! 2、参数模板化 (函数模板配合类模板) 较为复杂
template <class T1, class T2>
void printPerson2(Person<T1, T2> &p)
{
  p.showPerson();
  cout << "T1的类型为： " << typeid(T1).name() << endl; // todo typeid()可以获取变量的类型 .name()可以获取类型的字符串
  cout << "T2的类型为： " << typeid(T2).name() << endl;
}
void test02()
{
  Person<string, int> p("猪八戒", 90);
  printPerson2(p);
}

// ! 3、整个类模板化 (函数模板配合类模板) 较为复杂
template <class T>
void printPerson3(T &p)
{
  cout << "T的类型为： " << typeid(T).name() << endl;
  p.showPerson();
}
void test03()
{
  Person<string, int> p("唐僧", 30);
  printPerson3(p);
}

int main()
{

  test01();
  test02();
  test03();

  system("pause");

  return 0;
}
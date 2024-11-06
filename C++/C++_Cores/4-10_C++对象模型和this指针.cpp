#include <iostream>
using namespace std;

/*
!成员变量和成员函数分开存储
  *在C++中，类内的成员变量和成员函数分开存储
  *只有非静态成员变量才属于类的对象上
 */

// !空对象占用内存空间为：1
// *这是因为C++编译器会给每个空对象分配一个字节空间，是为了区分空对象占用的内存区域
// *每一个空对象也应该有一个独一无二的内存地址

class Person
{
public:
  Person()
  {
    mA = 0;
  }
  // !非静态成员变量占对象空间：4 属于类的对象上
  int mA;
  // !静态成员变量不占对象空间  不属于类的对象上
  static int mB;
  // !非静态成员函数也不占对象空间，所有函数共享一个函数实例  不属于类的对象上
  void func()
  {
    cout << "mA:" << this->mA << endl;
  }
  // !静态成员函数也不占对象空间  不属于类的对象上
  static void sfunc()
  {
  }
};

/*
!this指针
!c++通过提供特殊的对象指针，this指针，解决上述问题。this指针指向被调用的成员函数所属的对象
todo this指针是隐含每一个非静态成员函数内的一种指针
todo this指针不需要定义，直接使用即可
?this指针的用途：
  *  当形参和成员变量同名时，可用this指针来区分
  *  在类的非静态成员函数中返回对象本身，可使用return *this
 */

class Person2
{
public:
  Person2(int age)
  {
    // !1、当形参和成员变量同名时，可用this指针来区分
    // !this指针指向 被调用的成员函数所属的对象
    this->age = age;
  }

  // 返回本身要用引用：Person2 &  如果是Person2  则返回的是值，会调用拷贝构造函数，复制一份新的数据，就不是之前的对象了
  Person2 &PersonAddPerson(Person2 p)
  {
    this->age += p.age;
    // !2、返回对象本体用*this
    return *this;
  }

  int age;
};

void test01()
{
  Person2 p1(10);
  cout << "p1.age = " << p1.age << endl;

  Person2 p2(10);
  p2.PersonAddPerson(p1).PersonAddPerson(p1).PersonAddPerson(p1);
  cout << "p2.age = " << p2.age << endl;
}

int main()
{

  cout << sizeof(Person) << endl;

  test01();

  system("pause");

  return 0;
}
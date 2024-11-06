#include <iostream>
using namespace std;

/*
! 运算符重载概念：对已有的运算符重新进行定义，赋予其另一种功能，以适应不同的数据类型 例如类与对象
 */

/*
! 加号运算符重载
! 作用：实现两个自定义数据类型相加的运算
! 注意：不要滥用运算符重载
  */

class Person
{
public:
  Person() {};
  Person(int a, int b)
  {
    this->m_A = a;
    this->m_B = b;
  }
  // todo 1、成员函数实现 + 号运算符重载
  Person operator+(const Person &p)
  {
    Person temp;
    temp.m_A = this->m_A + p.m_A;
    temp.m_B = this->m_B + p.m_B;
    return temp;
  }

public:
  int m_A;
  int m_B;
};

// todo 2、全局函数实现 + 号运算符重载
// Person operator+(const Person& p1, const Person& p2) {
//	Person temp(0, 0);
//	temp.m_A = p1.m_A + p2.m_A;
//	temp.m_B = p1.m_B + p2.m_B;
//	return temp;
// }

// todo 3、运算符重载 可以发生函数重载
Person operator+(const Person &p2, int val)
{
  Person temp;
  temp.m_A = p2.m_A + val;
  temp.m_B = p2.m_B + val;
  return temp;
}

void test()
{

  Person p1(10, 10);
  Person p2(20, 20);

  // todo 成员函数方式
  Person p3 = p2 + p1; // ? 相当于 p2.operaor+(p1) 成员函数本质调用
  cout << "mA:" << p3.m_A << " mB:" << p3.m_B << endl;

  // todo 全局函数方式
  Person p4 = p1 + p2; // ? 相当于 operator+(p1,p2) 全局函数本质调用
  cout << "mA:" << p4.m_A << " mB:" << p4.m_B << endl;

  Person p5 = p3 + 10; // ? 相当于 operator+(p3,10) 函数重载的版本
  cout << "mA:" << p5.m_A << " mB:" << p5.m_B << endl;
}

int main()
{

  test();

  system("pause");

  return 0;
}
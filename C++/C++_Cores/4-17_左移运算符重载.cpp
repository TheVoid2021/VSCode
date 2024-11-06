#include <iostream>
using namespace std;

/*
! 左移运算符重载
! 作用：可以输出自定义数据类型
! 通过重载<<运算符，我们可以方便地将自定义类型的对象输出到控制台或其他输出流中，
! 而不需要手动调用对象的某个成员函数来输出数据。
 */

class Person
{
  friend ostream &operator<<(ostream &out, Person &p);

public:
  Person(int a, int b)
  {
    this->m_A = a;
    this->m_B = b;
  }

  // 成员函数 实现不了 左移运算符重载 p << cout 不是我们想要的效果
  // void operator<<(Person& p){
  // }

private:
  int m_A;
  int m_B;
};

// !只能用全局函数实现左移重载
// !cout是一个标准的输出流对象 即ostream对象 只能有一个
ostream &operator<<(ostream &out, Person &p) // out是引用传递
{
  out << "a:" << p.m_A << " b:" << p.m_B;
  return out;
}

void test()
{

  Person p1(10, 20);

  cout << p1 << "hello world" << endl; // !链式编程 所以?输入必须是返回值是ostream对象
}

int main()
{

  test();

  system("pause");

  return 0;
}
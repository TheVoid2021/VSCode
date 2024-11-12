#include <iostream>
using namespace std;

/*
! 继承中构造和析构顺序
! 子类继承父类后，当创建子类对象，也会调用父类的构造函数
! 继承中 先调用父类构造函数，再调用子类构造函数，析构顺序与构造相反
 */

class Base
{
public:
  Base()
  {
    cout << "base 构造函数" << endl;
  }
  ~Base()
  {
    cout << "base 析构函数" << endl;
  }
};

class son : public Base
{
public:
  son()
  {
    cout << "son 构造函数" << endl;
  }
  ~son()
  {
    cout << "son 析构函数" << endl;
  }
};

void test()
{
  son s;
}

int main()
{
  test();

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

/*
! 父类中所有非静态成员属性都会被子类继承下去，并且编译器会在子类的内部默认生成对应的构造函数
! 父类中私有成员属性也是被子类继承下去了，只是由编译器给隐藏后访问不到
 */

class Base
{
public:
  int m_A;

protected:
  int m_B;

private:
  int m_C; // !私有成员只是被隐藏了，但是还是会继承下去
};

// 公共继承
class Son : public Base
{
public:
  int m_D;
};

void test01()
{
  cout << "sizeof Son = " << sizeof(Son) << endl; // 16  继承三个 自身一个
}

int main()
{
  test01();

  system("pause");

  return 0;
}
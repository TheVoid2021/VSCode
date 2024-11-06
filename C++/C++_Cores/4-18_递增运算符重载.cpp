#include <iostream>
using namespace std;

/*
! 递增运算符重载
! 作用：通过重载递增运算符，实现自定义的递增运算，实现自己的整型数据
! 前置递增返回引用，后置递增返回值
 */

class MyInteger
{
  friend ostream &operator<<(ostream &out, MyInteger myint);

public:
  MyInteger()
  {
    m_Num = 0;
  }
  // todo 重载前置++运算符
  MyInteger &operator++() // !返回引用是为了实现连续递增，让本身跟着递增，即对同一个数据进行递增，而不是每次递增都创建一个新的对象
  {
    // todo 先进行++运算
    m_Num++;
    // todo 再将自身进行返回
    return *this;
  }

  // todo 重载后置++运算符
  MyInteger operator++(int) // ?int代表占位参数，只是为了区分前置和后置，没有实际意义
  {
    // todo 先记录当前结果
    MyInteger temp = *this; // ?记录当前本身的值，然后让本身的值加1，但是返回的是以前的值，达到先返回后++；
    // todo 递增
    m_Num++;
    // ! 最后返回记录的值 而不是引用，是因为temp是一个局部变量，函数执行完毕，temp就会被释放，返回引用就会报错
    return temp;
  }

private:
  int m_Num;
};

// *重载<<运算符
ostream &operator<<(ostream &out, MyInteger myint)
{
  out << myint.m_Num;
  return out;
}

// 前置++ 先++ 再返回
void test01()
{
  MyInteger myInt;
  cout << ++myInt << endl;
  cout << myInt << endl;
}

// 后置++ 先返回 再++
void test02()
{

  MyInteger myInt;
  cout << myInt++ << endl;
  cout << myInt << endl;
}

int main()
{

  test01();
  test02();

  system("pause");

  return 0;
}
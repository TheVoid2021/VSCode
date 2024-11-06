#include <iostream>
using namespace std;

/*
! 赋值运算符重载
todo c++编译器至少给一个类添加4个函数
  * 1. 默认构造函数(无参，函数体为空)
  * 2. 默认析构函数(无参，函数体为空)
  * 3. 默认拷贝构造函数，对属性进行值拷贝
  * 4. 赋值运算符 operator=, 对属性进行值拷贝
? 如果类中有属性指向堆区，做赋值操作时也会出现深浅拷贝问题
 */

class Person
{
public:
  Person(int age)
  {
    // 将年龄数据开辟到堆区
    m_Age = new int(age);
  }

  // todo 重载赋值运算符
  Person &operator=(Person &p) // ?这里用引用是为了避免拷贝构造函数的调用
  {
    if (m_Age != NULL) // todo 先判断是否有属性在堆区，如果有，则释放干净，然后再进行深拷贝
    {
      delete m_Age;
      m_Age = NULL;
    }
    // 编译器提供的代码是浅拷贝
    // m_Age = p.m_Age;

    // todo 提供深拷贝 解决浅拷贝的问题
    m_Age = new int(*p.m_Age);

    // todo 返回自身 为了实现连续赋值
    return *this;
  }

  ~Person()
  {
    // todo ?堆区开辟的内存，进行手动释放 但是赋值操作时，由于有两份地址指向，浅拷贝会导致堆区内存重复释放
    if (m_Age != NULL)
    {
      delete m_Age;
      m_Age = NULL;
    }
  }

  // 年龄的指针
  int *m_Age;
};

void test01()
{
  Person p1(18);

  Person p2(20);

  Person p3(30);

  p3 = p2 = p1; // 赋值操作

  cout << "p1的年龄为：" << *p1.m_Age << endl;

  cout << "p2的年龄为：" << *p2.m_Age << endl;

  cout << "p3的年龄为：" << *p3.m_Age << endl;
}

int main()
{

  test01();

  // int a = 10;
  // int b = 20;
  // int c = 30;

  // c = b = a;
  // cout << "a = " << a << endl;
  // cout << "b = " << b << endl;
  // cout << "c = " << c << endl;

  system("pause");

  return 0;
}
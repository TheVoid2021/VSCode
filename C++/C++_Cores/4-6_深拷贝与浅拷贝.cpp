#include <iostream>
using namespace std;

/*
!深拷贝与浅拷贝
  todo浅拷贝：简单的赋值拷贝操作
  todo深拷贝：在堆区重新申请空间，进行拷贝操作

!总结：如果属性有在堆区开辟的，一定要自己提供拷贝构造函数，防止浅拷贝带来的问题
 */

class Person
{
public:
  // 无参（默认）构造函数
  Person()
  {
    cout << "无参构造函数!" << endl;
  }
  // 有参构造函数
  Person(int age, int height)
  {

    cout << "有参构造函数!" << endl;

    m_age = age;
    m_height = new int(height);
  }
  // todo自己实现拷贝构造函数，解决浅拷贝带来的问题
  Person(const Person &p)
  {
    cout << "拷贝构造函数!" << endl;
    // 如果不利用深拷贝在堆区创建新内存，会导致浅拷贝带来的重复释放堆区问题
    m_age = p.m_age;
    // m_height = p.m_height; // !浅拷贝,编译器默认实现这行代码
    m_height = new int(*p.m_height); // !深拷贝
  }

  // 析构函数
  ~Person()
  {
    cout << "析构函数!" << endl;
    // 析构代码，将堆区开辟的数据进行释放操作
    if (m_height != NULL) // 如果没有释放堆区，则会出现内存泄漏
    {
      delete m_height;
      m_height = NULL;
    }
  }

public:
  int m_age;
  int *m_height;
};

void test01()
{
  Person p1(18, 180);

  Person p2(p1);

  cout << "p1的年龄： " << p1.m_age << " 身高： " << *p1.m_height << endl;

  cout << "p2的年龄： " << p2.m_age << " 身高： " << *p2.m_height << endl;
}

int main()
{

  test01();

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

// !空指针访问成员函数
class Person
{
public:
  void ShowClassName()
  {
    cout << "我是Person类!" << endl;
  }

  void ShowPerson()
  {
    // cout << "年龄为：" << mAge << endl; // 等价于this->mAge，默认的，this指向当前对象，下面传入的是空指针，所以会报错
    // !如果用到this指针，需要先加以判断保证代码的健壮性
    if (this == NULL)
    {
      return;
    }
    cout << mAge << endl;
  }

public:
  int mAge;
};

void test01()
{
  Person *p = NULL;
  p->ShowClassName(); // !空指针，可以调用成员函数
  p->ShowPerson();    // 但是如果成员函数中用到了this指针，就不可以了，需要加以判断
}

int main()
{

  test01();

  system("pause");

  return 0;
}
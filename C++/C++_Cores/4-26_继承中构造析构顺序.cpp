#include <iostream>
using namespace std;

/*
! �̳��й��������˳��
! ����̳и���󣬵������������Ҳ����ø���Ĺ��캯��
! �̳��� �ȵ��ø��๹�캯�����ٵ������๹�캯��������˳���빹���෴
 */

class Base
{
public:
  Base()
  {
    cout << "base ���캯��" << endl;
  }
  ~Base()
  {
    cout << "base ��������" << endl;
  }
};

class son : public Base
{
public:
  son()
  {
    cout << "son ���캯��" << endl;
  }
  ~son()
  {
    cout << "son ��������" << endl;
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
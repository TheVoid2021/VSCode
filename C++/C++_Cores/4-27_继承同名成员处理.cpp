#include <iostream>
using namespace std;

/*
! �̳�ͬ����Ա����
! ��������ͬ����Ա   ֱ�ӷ��ʼ���
! ���ʸ���ͬ����Ա   ��Ҫ��������  ����::����
! �������븸��ӵ��ͬ���ĳ�Ա��������������ظ�����ͬ����Ա����������������Է��ʵ�������ͬ������
 */

class Base
{
public:
  Base()
  {
    m_A = 100;
  }

  void func()
  {
    cout << "Base - func()����" << endl;
  }

  void func(int a)
  {
    cout << "Base - func(int a)����" << a << endl;
  }

public:
  int m_A;
};

class Son : public Base
{
public:
  Son()
  {
    m_A = 200;
  }

  // todo �������븸��ӵ��ͬ���ĳ�Ա��������������ظ��������а汾��ͬ����Ա����
  void func()
  {
    cout << "Son - func()����" << endl;
  }

public:
  int m_A;
};

void test01()
{
  Son s;

  cout << "Son�µ�m_A = " << s.m_A << endl;
  cout << "Base�µ�m_A = " << s.Base::m_A << endl; // todo �����ͨ�����������ʸ����е�ͬ����Ա���ԣ���Ҫ�Ӹ����������

  s.func();
  s.Base::func(); // todo �������ʸ����б����ص�ͬ����Ա��������Ҫ�Ӹ����������
  s.Base::func(10);
}
int main()
{

  test01();

  system("pause");
  return EXIT_SUCCESS;
}
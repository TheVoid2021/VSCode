#include <iostream>
using namespace std;

/*
! ���麯���ͳ�����
! ��̬�У�ͨ���������麯����ʵ���Ǻ�������ģ���Ҫ���ǵ���������д�����ݣ���˿��Խ��麯����Ϊ ���麯��
! ���麯���﷨��`virtual ����ֵ���� ������ �������б�= 0 ;`
! ���������˴��麯���������Ҳ��Ϊ =������=
? �������ص�**��
  * �޷�ʵ��������
  * ���������д�������еĴ��麯��������Ҳ���ڳ�����
 */

class Base // ! ������
{
public:
  // todo ���麯��
  // todo ����ֻҪ��һ�����麯���ͳ�Ϊ������
  // todo �������޷�ʵ��������
  // todo ���������д�����еĴ��麯��������Ҳ���ڳ�����
  virtual ~Base() {}       // ! �����������Ҳ����Ϊ�麯��
  virtual void func() = 0; // ! ���麯��
};

class Son : public Base
{
public:
  virtual void func() //!  ���������д�����еĴ��麯��������Ҳ���ڳ�����
  {
    cout << "func����" << endl;
  };
};

void test01()
{
  Base *base = NULL;
  // base = new Base; // ! ���󣬳������޷�ʵ��������
  base = new Son;
  base->func();
  delete base; // �ǵ�����
}

int main()
{

  test01();

  system("pause");

  return 0;
}
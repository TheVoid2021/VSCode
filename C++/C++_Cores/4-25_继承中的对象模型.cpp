#include <iostream>
using namespace std;

/*
! ���������зǾ�̬��Ա���Զ��ᱻ����̳���ȥ�����ұ���������������ڲ�Ĭ�����ɶ�Ӧ�Ĺ��캯��
! ������˽�г�Ա����Ҳ�Ǳ�����̳���ȥ�ˣ�ֻ���ɱ����������غ���ʲ���
 */

class Base
{
public:
  int m_A;

protected:
  int m_B;

private:
  int m_C; // !˽�г�Աֻ�Ǳ������ˣ����ǻ��ǻ�̳���ȥ
};

// �����̳�
class Son : public Base
{
public:
  int m_D;
};

void test01()
{
  cout << "sizeof Son = " << sizeof(Son) << endl; // 16  �̳����� ����һ��
}

int main()
{
  test01();

  system("pause");

  return 0;
}
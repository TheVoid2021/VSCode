#include <iostream>
using namespace std;

/*
!��̬��Ա�����ڳ�Ա�����ͳ�Ա����ǰ���Ϲؼ���static����Ϊ��̬��Ա
?��̬��Ա��Ϊ��
  todo��̬��Ա����
   *  ���ж�����ͬһ������
   *  �ڱ���׶η����ڴ� ȫ����
   *  ���������������ʼ��
  todo��̬��Ա����
   *  ���ж�����ͬһ������
   *  ��̬��Ա����ֻ�ܷ��ʾ�̬��Ա����
 */

class Person
{
public:
  // todo��������
  static int m_A; // !��̬��Ա��������

  // ��̬��Ա�����ص㣺
  // 1 �ڱ���׶η����ڴ�
  // 2 ���������������ʼ��
  // 3 ���ж�����ͬһ������

  // ��̬��Ա�����ص㣺
  // 1 ������һ������
  // 2 ��̬��Ա����ֻ�ܷ��ʾ�̬��Ա����

  static void func() // !��̬��Ա��������
  {
    cout << "func����" << endl;
    m_C = 100;
    // m_D = 100; //!���󣬲����Է��ʷǾ�̬��Ա����
  }

  static int m_C; // ��̬��Ա����
  int m_D;        // �Ǿ�̬��Ա����

private:
  // todo��������
  static int m_B; // ?��̬��Ա����Ҳ���з���Ȩ�޵�

  // ?��̬��Ա����Ҳ���з���Ȩ�޵�
  static void func2()
  {
    cout << "func2����" << endl;
  }
};
int Person::m_A = 10; // todo�����ʼ��
int Person::m_B = 10; // todo�����ʼ��

int Person::m_C = 10; // todo�����ʼ��

void test01()
{
  // !��̬��Ա�������ַ��ʷ�ʽ

  // todo 1��ͨ������
  Person p1;
  p1.m_A = 100;
  cout << "p1.m_A = " << p1.m_A << endl;

  Person p2;
  p2.m_A = 200;
  cout << "p1.m_A = " << p1.m_A << endl; // !����ͬһ������
  cout << "p2.m_A = " << p2.m_A << endl;

  // todo 2��ͨ������
  cout << "m_A = " << Person::m_A << endl;

  // cout << "m_B = " << Person::m_B << endl; // ?˽��Ȩ�޷��ʲ���

  // !��̬��Ա�������ַ��ʷ�ʽ

  // todo 1��ͨ������
  Person p3;
  p3.func();

  // todo 2��ͨ������
  Person::func();

  // Person::func2(); // ?˽��Ȩ�޷��ʲ���
}

int main()
{

  test01();

  system("pause");

  return 0;
}
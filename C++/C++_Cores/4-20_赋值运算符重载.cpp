#include <iostream>
using namespace std;

/*
! ��ֵ���������
todo c++���������ٸ�һ�������4������
  * 1. Ĭ�Ϲ��캯��(�޲Σ�������Ϊ��)
  * 2. Ĭ����������(�޲Σ�������Ϊ��)
  * 3. Ĭ�Ͽ������캯���������Խ���ֵ����
  * 4. ��ֵ����� operator=, �����Խ���ֵ����
? �������������ָ�����������ֵ����ʱҲ�������ǳ��������
 */

class Person
{
public:
  Person(int age)
  {
    // ���������ݿ��ٵ�����
    m_Age = new int(age);
  }

  // todo ���ظ�ֵ�����
  Person &operator=(Person &p) // ?������������Ϊ�˱��⿽�����캯���ĵ���
  {
    if (m_Age != NULL) // todo ���ж��Ƿ��������ڶ���������У����ͷŸɾ���Ȼ���ٽ������
    {
      delete m_Age;
      m_Age = NULL;
    }
    // �������ṩ�Ĵ�����ǳ����
    // m_Age = p.m_Age;

    // todo �ṩ��� ���ǳ����������
    m_Age = new int(*p.m_Age);

    // todo �������� Ϊ��ʵ��������ֵ
    return *this;
  }

  ~Person()
  {
    // todo ?�������ٵ��ڴ棬�����ֶ��ͷ� ���Ǹ�ֵ����ʱ�����������ݵ�ַָ��ǳ�����ᵼ�¶����ڴ��ظ��ͷ�
    if (m_Age != NULL)
    {
      delete m_Age;
      m_Age = NULL;
    }
  }

  // �����ָ��
  int *m_Age;
};

void test01()
{
  Person p1(18);

  Person p2(20);

  Person p3(30);

  p3 = p2 = p1; // ��ֵ����

  cout << "p1������Ϊ��" << *p1.m_Age << endl;

  cout << "p2������Ϊ��" << *p2.m_Age << endl;

  cout << "p3������Ϊ��" << *p3.m_Age << endl;
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
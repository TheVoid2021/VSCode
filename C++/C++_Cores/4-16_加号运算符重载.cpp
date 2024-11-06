#include <iostream>
using namespace std;

/*
! ��������ظ�������е���������½��ж��壬��������һ�ֹ��ܣ�����Ӧ��ͬ���������� �����������
 */

/*
! �Ӻ����������
! ���ã�ʵ�������Զ�������������ӵ�����
! ע�⣺��Ҫ�������������
  */

class Person
{
public:
  Person() {};
  Person(int a, int b)
  {
    this->m_A = a;
    this->m_B = b;
  }
  // todo 1����Ա����ʵ�� + �����������
  Person operator+(const Person &p)
  {
    Person temp;
    temp.m_A = this->m_A + p.m_A;
    temp.m_B = this->m_B + p.m_B;
    return temp;
  }

public:
  int m_A;
  int m_B;
};

// todo 2��ȫ�ֺ���ʵ�� + �����������
// Person operator+(const Person& p1, const Person& p2) {
//	Person temp(0, 0);
//	temp.m_A = p1.m_A + p2.m_A;
//	temp.m_B = p1.m_B + p2.m_B;
//	return temp;
// }

// todo 3����������� ���Է�����������
Person operator+(const Person &p2, int val)
{
  Person temp;
  temp.m_A = p2.m_A + val;
  temp.m_B = p2.m_B + val;
  return temp;
}

void test()
{

  Person p1(10, 10);
  Person p2(20, 20);

  // todo ��Ա������ʽ
  Person p3 = p2 + p1; // ? �൱�� p2.operaor+(p1) ��Ա�������ʵ���
  cout << "mA:" << p3.m_A << " mB:" << p3.m_B << endl;

  // todo ȫ�ֺ�����ʽ
  Person p4 = p1 + p2; // ? �൱�� operator+(p1,p2) ȫ�ֺ������ʵ���
  cout << "mA:" << p4.m_A << " mB:" << p4.m_B << endl;

  Person p5 = p3 + 10; // ? �൱�� operator+(p3,10) �������صİ汾
  cout << "mA:" << p5.m_A << " mB:" << p5.m_B << endl;
}

int main()
{

  test();

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

/*
! �������������
! ���ã���������Զ�����������
! ͨ������<<����������ǿ��Է���ؽ��Զ������͵Ķ������������̨������������У�
! ������Ҫ�ֶ����ö����ĳ����Ա������������ݡ�
 */

class Person
{
  friend ostream &operator<<(ostream &out, Person &p);

public:
  Person(int a, int b)
  {
    this->m_A = a;
    this->m_B = b;
  }

  // ��Ա���� ʵ�ֲ��� ������������� p << cout ����������Ҫ��Ч��
  // void operator<<(Person& p){
  // }

private:
  int m_A;
  int m_B;
};

// !ֻ����ȫ�ֺ���ʵ����������
// !cout��һ����׼����������� ��ostream���� ֻ����һ��
ostream &operator<<(ostream &out, Person &p) // out�����ô���
{
  out << "a:" << p.m_A << " b:" << p.m_B;
  return out;
}

void test()
{

  Person p1(10, 20);

  cout << p1 << "hello world" << endl; // !��ʽ��� ����?��������Ƿ���ֵ��ostream����
}

int main()
{

  test();

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

/*
!const���γ�Ա����
todo��������const + ��Ա����
  * ��Ա�������const�����ǳ�Ϊ�������Ϊ ������
  ? �������ڲ������޸ĳ�Ա����
  * ��Ա��������ʱ�ӹؼ���mutable���ڳ���������Ȼ�����޸�
todo������const + ����
  * ��������ǰ��const�Ƹö���Ϊ������
  ? ������ֻ�ܵ��ó�����
 */

class Person
{
public:
  Person()
  {
    m_A = 0;
    m_B = 0;
  }

  // !thisָ��ı�����һ��ָ�볣����ָ���ָ�򲻿��޸�
  // !�������ָ��ָ���ֵҲ�������޸ģ���Ҫ����������?
  void ShowPerson() const // !const���γ�Ա��������ʾָ��ָ����ڴ�ռ������Ҳ�����޸ģ�����mutable���εı���
  {
    // const Type* const pointer;
    // this = NULL; //!�����޸�ָ���ָ�� Ĭ��Person* const this;
    // this->mA = 100; //!����thisָ��ָ��Ķ���������ǿ����޸ĵ�

    this->m_B = 100;
  }

  void MyFunc() const
  {
    // mA = 10000;
  }

public:
  int m_A;
  mutable int m_B; // todo ���޸� �ɱ��
};

// const���ζ���  ������
void test01()
{

  const Person person; // !��������
  cout << person.m_A << endl;
  // person.mA = 100; // !���������޸ĳ�Ա������ֵ,���ǿ��Է���?
  person.m_B = 100; // !���ǳ���������޸�mutable���γ�Ա����

  // !������ֻ�ܵ��ó����� ���޶���һ��ֻ��״̬?
  person.MyFunc(); // !�������ܵ�����ͨ��Ա��������Ϊ��ͨ��Ա���������޸ĳ�Ա������ֵ
}

int main()
{

  test01();

  system("pause");

  return 0;
}
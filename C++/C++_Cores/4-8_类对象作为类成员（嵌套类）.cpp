#include <iostream>
using namespace std;

/*
!C++���еĳ�Ա��������һ����Ķ������ǳƸó�ԱΪ �����Ա
?B�����ж���A��Ϊ��Ա,AΪ�����Ա,BΪ�ⲿ��
*��ô������B����ʱ��A��B�Ĺ����������˳����˭��˭��
 */

class Phone
{
public:
  Phone(string name)
  {
    m_PhoneName = name;
    cout << "Phone����" << endl;
  }

  ~Phone()
  {
    cout << "Phone����" << endl;
  }

  string m_PhoneName;
};

class Person
{
public:
  // !��ʼ���б���Ը��߱�����������һ�����캯��
  // Phone m_Phone = pName; ��ʽת����
  Person(string name, string pName) : m_Name(name), m_Phone(pName)
  {
    cout << "Person����" << endl;
  }

  ~Person()
  {
    cout << "Person����" << endl;
  }

  void playGame()
  {
    cout << m_Name << " ʹ��" << m_Phone.m_PhoneName << " ���ֻ�! " << endl;
  }

  string m_Name;
  Phone m_Phone; // todo�����Ա
};
void test01()
{
  // !�����г�Ա�����������ʱ�����ǳƸó�ԱΪ �����Ա
  // !�����˳���� ���ȵ��ö����Ա�Ĺ��죬�ٵ��ñ��๹��
  // !����˳���빹���෴
  Person p("����", "ƻ��X");
  p.playGame();
}

int main()
{

  test01();

  system("pause");

  return 0;
}
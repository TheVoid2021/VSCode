#include <iostream>
using namespace std;

/*
! ��ģ���Ա��������ʵ��
* ��ģ���г�Ա��������ʵ��ʱ����Ҫ����ģ������б�!

? ���ʣ����캯���ͳ�Ա������ʲô����
  * 1.���캯��û�з���ֵ���ͣ���Ա�����з���ֵ����
  * 2.���캯���ڶ��󴴽���ʱ���Զ����ã���Ա�����ڶ��󴴽�֮�����
  * 3.���캯���ĺ�������������ͬ����Ա�����ĺ�������������ͬ
  * 4.���캯��ֻ����һ������Ա���������ж��
  * 5.���캯���Ĳ����б����û�У���Ա�����Ĳ����б������Ҳ����û��
  * 6.���캯���Ĳ����б������Ĭ��ֵ����Ա�����Ĳ����б������Ҳ����û��
 */

// ��ģ���г�Ա��������ʵ��
template <class T1, class T2> // classҲ������typename����
class Person
{
public:
  // todo ��Ա������������
  Person(T1 name, T2 age);
  void showPerson();

public:
  T1 m_Name;
  T2 m_Age;
};

// todo ���캯�� ����ʵ��
template <class T1, class T2>
Person<T1, T2>::Person(T1 name, T2 age) // Person<T1, T2>��ʾ����һ����ģ��Ĺ��캯��
{
  this->m_Name = name;
  this->m_Age = age;
}

// todo ��Ա���� ����ʵ��
template <class T1, class T2>
void Person<T1, T2>::showPerson()
{
  cout << "����: " << this->m_Name << " ����:" << this->m_Age << endl;
}

void test01()
{
  Person<string, int> p("Tom", 20);
  p.showPerson();
}

int main()
{

  test01();

  system("pause");

  return 0;
}
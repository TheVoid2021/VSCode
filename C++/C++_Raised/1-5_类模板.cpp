#include <iostream>
using namespace std;

/*
! ��ģ��ͺ���ģ���﷨���ƣ�������ģ��template������࣬�����Ϊ��ģ��
! ��ģ���﷨
? ��ģ�����ã�
  * ����һ��ͨ���࣬���еĳ�Ա �������Ϳ��Բ������ƶ�����һ��**���������**������
? �﷨��
  * template<typename T>
  * ��
? ���ͣ�
  * template  ---  ��������ģ��
  * typename  --- ���������ķ�����һ���������ͣ�������class����
  * T    ---   ͨ�õ��������ͣ����ƿ����滻��ͨ��Ϊ��д��ĸ
 */

/*
! ��ģ���뺯��ģ������
? ��ģ���뺯��ģ��������Ҫ�����㣺
  * 1. ��ģ��û���Զ������Ƶ���ʹ�÷�ʽ��ֻ������ʾָ������
  * 2. ��ģ����ģ������б��п�����Ĭ�ϲ���
 */

/*
! ��ģ���г�Ա��������ʱ��
? ��ģ���г�Ա��������ͨ���г�Ա��������ʱ����������ģ�
  * ��ͨ���еĳ�Ա����һ��ʼ�Ϳ��Դ���
  * ��ģ���еĳ�Ա�����ڵ���ʱ�Ŵ���
 */

// ! ��ģ��
template <class NameType, class AgeType = int> // todo ��ģ���п��Զ���Ĭ�ϲ���
class Person
{
public:
  Person(NameType name, AgeType age)
  {
    this->mName = name;
    this->mAge = age;
  }
  void showPerson()
  {
    cout << "name: " << this->mName << " age: " << this->mAge << endl;
  }

public:
  NameType mName;
  AgeType mAge;
};

void test01()
{
  // ָ��NameType Ϊstring���ͣ�AgeType Ϊ int����
  Person<string, int> P1("�����", 999);
  P1.showPerson();
}

// todo 1����ģ��û���Զ������Ƶ���ʹ�÷�ʽ
void test02()
{
  // Person p("�����", 1000); // ���� ��ģ��ʹ��ʱ�򣬲��������Զ������Ƶ�
  Person<string, int> p("�����", 1000); // ����ʹ����ʾָ�����͵ķ�ʽ��ʹ����ģ��
  p.showPerson();
}

// todo 2����ģ����ģ������б��п�����Ĭ�ϲ���
void test03()
{
  Person<string> p("��˽�", 999); // ��ģ���е�ģ������б� ����ָ��Ĭ�ϲ���
  p.showPerson();
}

class Person1
{
public:
  void showPerson1()
  {
    cout << "Person1 showPerson1" << endl;
  }
};

class Person2
{
public:
  void showPerson2()
  {
    cout << "Person2 showPerson2" << endl;
  }
};

template <class T>
class Myclass
{
public:
  T obj;

  // ! ��ģ���еĳ�Ա����������һ��ʼ�ʹ����ģ������ڵ��õ�ʱ��Ŵ���
  void func1()
  {
    obj.showPerson1();
  }

  void func2()
  {
    obj.showPerson2();
  }
};

void test01()
{
  Myclass<Person1> m1;
  m1.func1();
  m1.func2();
}

int main()
{

  test01();

  test02();

  test03();

  system("pause");

  return 0;
}
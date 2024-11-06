#include <iostream>
using namespace std;

/*
!��װ�������Ȩ�޷�������
*�������ʱ�����԰����Ժ���Ϊ���ڲ�ͬ��Ȩ���£����Կ���

*����Ȩ�������֣�
todo 1. public       ����Ȩ��    ���ڿ��Է��ʣ�������Է���
todo 2. protected    ����Ȩ��    ���ڿ��Է��ʣ����ⲻ���Է��ʣ�������Է���
todo 3. private      ˽��Ȩ��    ���ڿ��Է��ʣ����ⲻ���Է��ʣ����಻���Է���
*/
class Person
{
public:
  // ����Ȩ��
  string m_Name; // ����

protected:
  // ����Ȩ��
  string m_Car; // ����

private:
  // ˽��Ȩ��
  int m_Password; // ���п�����

public:
  void func()
  {
    m_Name = "����";
    m_Car = "������";
    m_Password = 123456;
  }
};

/*
!��C++�� struct��classΨһ����������� Ĭ�ϵķ���Ȩ�޲�ͬ
*����
todo struct  Ĭ��Ȩ��Ϊ����
todo class   Ĭ��Ȩ��Ϊ˽�� */
class c1
{
  int m_A; // Ĭ����˽��Ȩ��
};

struct s1
{
  int m_A; // Ĭ���ǹ���Ȩ��
};

/*
!��Ա��������Ϊ˽��
*�ŵ�1�������г�Ա��������Ϊ˽�У������Լ����ƶ�дȨ��
*�ŵ�2������дȨ�ޣ����ǿ��Լ�����ݵ���Ч�� */
class Person1
{
public:
  // �������ÿɶ���д
  void setName(string name) // д
  {
    m_name = name;
  }
  string getName() // ��
  {
    return m_name;
  }

  // ��ȡ����
  int getAge()
  {
    return m_Age;
  }
  // ��������
  void setAge(int age)
  {
    if (age < 0 || age > 150)
    {
      cout << "���������!" << endl;
      return;
    }
    m_Age = age;
  }

  // ��������Ϊֻд
  void setLover(string lover)
  {
    m_Lover = lover;
  }

private:
  string m_name; // �ɶ���д  ����

  int m_Age; // ֻ��  ����

  string m_Lover; // ֻд  ����
};

//! ������
int main()
{
  Person p1;
  p1.m_Name = "����";
  // p1.m_Car = "����"; // ���󣬱���Ȩ�ޣ����ⲻ���Է���
  // p1.m_Password = 123456; // ����˽��Ȩ�ޣ����ⲻ���Է���
  p1.func();

  // c1 c;
  //  c.m_A = 100; // ����Ĭ����˽��Ȩ�ޣ����ⲻ���Է���

  // s1 s;
  // s.m_A = 100; // ��ȷ��Ĭ���ǹ���Ȩ�ޣ�������Է���

  Person1 p;
  // ��������
  p.setName("����");
  cout << "������ " << p.getName() << endl;

  // ��������
  p.setAge(50); // todoдȨ�ޣ�������ݵ���Ч��
  // p.m_Age = 100; // ����ֻ��Ȩ�ޣ����ⲻ���Է���
  cout << "���䣺 " << p.getAge() << endl;

  // ��������
  p.setLover("�Ծ�");
  // cout << "���ˣ� " << p.m_Lover << endl;  //ֻд���ԣ������Զ�ȡ
  return 0;
}
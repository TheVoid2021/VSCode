#include <iostream>
using namespace std;

/*
!��Ա�����ͳ�Ա�����ֿ��洢
  *��C++�У����ڵĳ�Ա�����ͳ�Ա�����ֿ��洢
  *ֻ�зǾ�̬��Ա������������Ķ�����
 */

// !�ն���ռ���ڴ�ռ�Ϊ��1
// *������ΪC++���������ÿ���ն������һ���ֽڿռ䣬��Ϊ�����ֿն���ռ�õ��ڴ�����
// *ÿһ���ն���ҲӦ����һ����һ�޶����ڴ��ַ

class Person
{
public:
  Person()
  {
    mA = 0;
  }
  // !�Ǿ�̬��Ա����ռ����ռ䣺4 ������Ķ�����
  int mA;
  // !��̬��Ա������ռ����ռ�  ��������Ķ�����
  static int mB;
  // !�Ǿ�̬��Ա����Ҳ��ռ����ռ䣬���к�������һ������ʵ��  ��������Ķ�����
  void func()
  {
    cout << "mA:" << this->mA << endl;
  }
  // !��̬��Ա����Ҳ��ռ����ռ�  ��������Ķ�����
  static void sfunc()
  {
  }
};

/*
!thisָ��
!c++ͨ���ṩ����Ķ���ָ�룬thisָ�룬����������⡣thisָ��ָ�򱻵��õĳ�Ա���������Ķ���
todo thisָ��������ÿһ���Ǿ�̬��Ա�����ڵ�һ��ָ��
todo thisָ�벻��Ҫ���壬ֱ��ʹ�ü���
?thisָ�����;��
  *  ���βκͳ�Ա����ͬ��ʱ������thisָ��������
  *  ����ķǾ�̬��Ա�����з��ض�������ʹ��return *this
 */

class Person2
{
public:
  Person2(int age)
  {
    // !1�����βκͳ�Ա����ͬ��ʱ������thisָ��������
    // !thisָ��ָ�� �����õĳ�Ա���������Ķ���
    this->age = age;
  }

  // ���ر���Ҫ�����ã�Person2 &  �����Person2  �򷵻ص���ֵ������ÿ������캯��������һ���µ����ݣ��Ͳ���֮ǰ�Ķ�����
  Person2 &PersonAddPerson(Person2 p)
  {
    this->age += p.age;
    // !2�����ض�������*this
    return *this;
  }

  int age;
};

void test01()
{
  Person2 p1(10);
  cout << "p1.age = " << p1.age << endl;

  Person2 p2(10);
  p2.PersonAddPerson(p1).PersonAddPerson(p1).PersonAddPerson(p1);
  cout << "p2.age = " << p2.age << endl;
}

int main()
{

  cout << sizeof(Person) << endl;

  test01();

  system("pause");

  return 0;
}
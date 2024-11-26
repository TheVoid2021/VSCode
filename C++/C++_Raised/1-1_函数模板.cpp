#include <iostream>
using namespace std;

/*
! ģ����ǽ��� *ͨ�õ�ģ��* �������� ������
? ģ����ص㣺
  * ģ�岻����ֱ��ʹ�ã���ֻ��һ�����
  * ģ���ͨ�ò��������ܵ�
todo C++��һ�ֱ��˼���Ϊ =���ͱ��= ����Ҫ���õļ�������ģ��
todo C++�ṩ����ģ�����: *����ģ��* �� *��ģ��*

? ����ģ�����ã�
  * ����һ��ͨ�ú������亯������ֵ���ͺ��β����Ϳ��Բ������ƶ�����һ�� *���������* ������
todo �﷨��
    * template<typename T>
    * ������������
todo ���ͣ�
    * template  ---  ��������ģ��
    * typename  --- ���������ķ�����һ���������ͣ�������class����
    * T    ---   ͨ�õ��������ͣ����ƿ����滻��ͨ��Ϊ��д��ĸ
 */

// �������ͺ���
void swapInt(int &a, int &b)
{
  int temp = a;
  a = b;
  b = temp;
}

// ���������ͺ���
void swapDouble(double &a, double &b)
{
  double temp = a;
  a = b;
  b = temp;
}

// ! ����ģ���ṩͨ�õĽ�������
template <typename T> // todo ����һ��ģ�壬���߱�������������н����ŵĴ������������Ͳ���T��ͨ�õ�
void mySwap(T &a, T &b)
{
  T temp = a;
  a = b;
  b = temp;
}

void test01()
{
  int a = 10;
  int b = 20;

  // swapInt(a, b);

  // todo ����ģ��ʵ�ֽ��� (���ַ�ʽ)
  // * 1���Զ������Ƶ�
  mySwap(a, b);

  // * 2����ʾָ������
  mySwap<int>(a, b);

  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
}

// todo ����ģ���ṩͨ�õĽ�������
template <class T> // todo ���������ķ�����һ���������ͣ�typename������class����
void mySwap1(T &a, T &b)
{
  T temp = a;
  a = b;
  b = temp;
}

// ! 1���Զ������Ƶ��������Ƶ���һ�µ���������T,�ſ���ʹ��
void test02()
{
  int a = 10;
  int b = 20;
  // char c = 'c';

  mySwap1(a, b); // ��ȷ�������Ƶ���һ�µ�T
                 // mySwap(a, c); // �����Ƶ�����һ�µ�T����
}

// ! 2��ģ�����Ҫȷ����T���������ͣ��ſ���ʹ��
template <class T>
void func()
{
  cout << "func ����" << endl;
}

void test03()
{
  // func(); //����ģ�岻�ܶ���ʹ�ã�����ȷ����T������
  func<int>(); // ������ʾָ�����͵ķ�ʽ����Tһ�����ͣ��ſ���ʹ�ø�ģ��
}

int main()
{

  test01();

  test02();

  test03();

  system("pause");

  return 0;
}
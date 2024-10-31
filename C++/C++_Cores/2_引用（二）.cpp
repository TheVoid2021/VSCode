#include <iostream>
using namespace std;

/*
!��������������

*���ã���������ʱ�������������õļ������β�����ʵ��

*�ŵ㣺���Լ�ָ���޸�ʵ�� */

/*
!��������������ֵ

*���ã������ǿ�����Ϊ�����ķ���ֵ���ڵ�

*ע�⣺��Ҫ���ؾֲ���������

*�÷�������������Ϊ��ֵ */

// !1. ֵ����
void mySwap01(int a, int b)
{
  int temp = a;
  a = b;
  b = temp;
}

// !2. ��ַ����
void mySwap02(int *a, int *b)
{
  int temp = *a;
  *a = *b;
  *b = temp;
}

// !3. ���ô���
void mySwap03(int &a, int &b)
{
  int temp = a;
  a = b;
  b = temp;
}

//! ���ؾֲ���������
// int &test01()
// {
//   int c = 10; // �ֲ�����
//   return c;
// }

//! ���ؾ�̬��������
int &test02()
{
  static int a = 20;
  return a;
}

//! ���õı�����c++�ڲ�ʵ����һ��ָ�볣��.
//  ���������ã�ת��Ϊ int* const ref = &a;
void func(int &ref)
{
  ref = 100; // ref�����ã�ת��Ϊ*ref = 100
}

//! ����������Ҫ���������βΣ���ֹ�����
//! �ں����β��б��У����Լ�const�����βΣ���ֹ�βθı�ʵ��
// ����ʹ�õĳ�����ͨ�����������β�
void showValue(const int &v)
{
  // v += 10;
  cout << v << endl;
}

int main()
{

  int a = 10;
  int b = 20;

  // todoֵ���ݣ��ββ�������ʵ��
  mySwap01(a, b);
  cout << "a:" << a << " b:" << b << endl;

  // todo��ַ���ݣ��βλ�����ʵ��
  mySwap02(&a, &b);
  cout << "a:" << a << " b:" << b << endl;

  // todo���ô��ݣ��βλ�����ʵ��
  mySwap03(a, b);
  cout << "a:" << a << " b:" << b << endl;

  // todo���ܷ��ؾֲ�����������
  // int &ref = test01();
  // cout << "ref = " << ref << endl; // ��һ�ν����ȷ������Ϊ���������˱���
  // cout << "ref = " << ref << endl; // �ڶ��ν��������Ϊ�ֲ�����a�Ѿ��ͷ�

  // todo�����������ֵ����ô���뷵������
  int &ref2 = test02();
  cout << "ref2 = " << ref2 << endl; // 0
  cout << "ref2 = " << ref2 << endl; //  test02() = 100;

  test02() = 1000;

  cout << "ref2 = " << ref2 << endl;
  cout << "ref2 = " << ref2 << endl;

  int c = 10;

  // �Զ�ת��Ϊ int* const ref = &a; ָ�볣����ָ��ָ�򲻿ɸģ�Ҳ˵��Ϊʲô���ò��ɸ���
  int &ref = c;
  ref = 20; // �ڲ�����ref�����ã��Զ�������ת��Ϊ: *ref = 20;

  cout << "c:" << c << endl;
  cout << "ref:" << ref << endl;

  func(c);

  // int& ref = 10;  ���ñ�����Ҫһ���Ϸ����ڴ�ռ䣬������д���
  // ����const�Ϳ����ˣ��������Ż����룬int temp = 10; const int& ref = temp;
  const int &ref1 = 10;

  // ref = 100;  //����const�󲻿����޸ı���
  cout << ref1 << endl;

  // ���������ó������÷�ֹ������޸�ʵ��
  int d = 10;
  showValue(d);

  system("pause");

  return 0;
}

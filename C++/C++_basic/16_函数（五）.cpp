#include <iostream>
using namespace std;

/*  �����ķ��ļ���д
���ã��ô���ṹ��������
�������ļ���дһ����4������
1. ������׺��Ϊ.h��ͷ�ļ�
2. ������׺��Ϊ.cpp��Դ�ļ�
3. ��ͷ�ļ���д����������
4. ��Դ�ļ���д�����Ķ��� */

// swap.h�ļ�
//  ʵ���������ֽ����ĺ�������
// void swap(int a, int b);

// swap.cpp�ļ�
// #include "swap.h"
/* void swap(int a, int b)
{
  int temp = a;
  a = b;
  b = temp;

  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
} */

// main�����ļ�
// #include "swap.h"
int main()
{

  int a = 100;
  int b = 200;
  swap(a, b);

  cout << "a = " << a << endl;

  system("pause");

  return 0;
}
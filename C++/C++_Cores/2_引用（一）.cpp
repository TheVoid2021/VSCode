#include <iostream>
using namespace std;

/*
!����
*���ã� ���������������&

*�﷨�� `�������� &���� = ԭ��` */

int main()
{

  int a = 10;
  int &b = a; // !һ����ʼ���󣬾Ͳ����Ը���
  int c = 20;

  // int &c; // !�������ñ����ʼ��

  b = c; // !���Ǹ�ֵ���������Ǹ������ã����ñ���û�п����ڴ�ռ䣬��ֻ��ԭ��������

  // ����ͬһ���ڴ�ռ䣬�޸�һ������һ��Ҳ��ı�
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "c = " << c << endl;

  b = 100;

  cout << "a = " << a << endl;
  cout << "b = " << b << endl;

  system("pause");

  return 0;
}
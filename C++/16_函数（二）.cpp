#include <iostream>
using namespace std;

/*
 * ֵ���ݣ����Ǻ�������ʱʵ�ν���ֵ������β�
 * ֵ����ʱ������βη����κθı䣬������Ӱ��ʵ��
 */

void swap(int num1, int num2)
{
  cout << "����ǰ��" << endl;
  cout << "num1 = " << num1 << endl;
  cout << "num2 = " << num2 << endl;

  int temp = num1;
  num1 = num2;
  num2 = temp;

  cout << "������" << endl;
  cout << "num1 = " << num1 << endl;
  cout << "num2 = " << num2 << endl;

  // return ; ����������ʱ�򣬲���Ҫ����ֵ�����Բ�дreturn
}

int main()
{

  int a = 10;
  int b = 20;

  swap(a, b);

  cout << "main�е� a ����= " << a << endl;
  cout << "main�е� b ����= " << b << endl;

  system("pause");

  return 0;
}
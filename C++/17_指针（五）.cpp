#include <iostream>
using namespace std;

// !ֵ����
void swap1(int a, int b)
{
  int temp = a;
  a = b;
  b = temp;
  //! ��������β�a��b���ˣ����������ʵ��a��b��û�б�
}
// !��ַ����
void swap2(int *p1, int *p2) // �ѵ�ַ����ָ��
{
  int temp = *p1;
  *p1 = *p2;
  *p2 = temp;
  //! �����p1��p2��ָ�룬ָ��ʵ�εĵ�ַ�����Ըı�ָ��ָ���ֵ����������β�p1��p2���ˣ������ʵ��a��bҲ����
}

int main()
{
  // *�ܽ᣺��������޸�ʵ�Σ�����ֵ���ݣ�������޸�ʵ�Σ����õ�ַ����
  int a = 10;
  int b = 20;
  swap1(a, b); // !ֵ���ݲ���ı�ʵ��

  swap2(&a, &b); // !��ַ���ݻ�ı�ʵ��

  cout << "a = " << a << endl;

  cout << "b = " << b << endl;

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

int main()
{ // main�������ҽ���һ��

  /*
   * C��C++���ַ��ͱ���ֻռ��һ���ֽ�
   * �ַ��ͱ��������ǵ������ַ�����ŵ��ڴ��д��棬���ǰ����Ӧ��ASCII������뵽���浥Ԫ
   */

  // 1���ַ��ͱ���������ʽ
  char ch = 'a';
  cout << ch << endl; // �ַ��ͱ���ֻ��һ���ַ�������ֻ���õ�����

  // 2���ַ��ͱ�����ռ�ڴ��С
  cout << "char�ַ��ͱ�����ռ�ڴ�:" << sizeof(char) << endl; // �ַ����Ϳ����ж���ַ�����Ҫ��˫����

  // 3���ַ��ͱ�����ӦASCII����
  cout << (int)ch << endl; // �鿴�ַ�a��Ӧ��ASCII��
  ch = 97;                 // ����ֱ����ASCII���ַ��ͱ�����ֵ
  cout << ch << endl;

  system("pause");

  return 0;
}
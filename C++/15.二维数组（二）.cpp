#include <iostream>
using namespace std;

int main()
{
  /*
    1.�鿴��ά������ռ�ڴ�ռ�
    2.��ȡ��ά�����׵�ַ
  */

  int arr[2][3] =
      {
          {1, 2, 3},
          {4, 5, 6},
      };
  // ��ַ
  cout << "��ά������ռ�ڴ�ռ�Ϊ��" << sizeof(arr) << endl;
  cout << "��ά�����׵�ַΪ��" << (int *)arr << endl;
  cout << "��ά�����һ���׵�ַΪ��" << (int *)arr[0] << endl;
  cout << "��ά����ڶ����׵�ַΪ��" << (int *)arr[1] << endl;
  cout << "��ά�����һ�е�һ��Ԫ�ص�ַΪ��" << (long long)arr[0][0] << endl;
  cout << "��ά����ڶ��е�һ��Ԫ�ص�ַΪ��" << (long long)arr[1][0] << endl;

  // ��С
  cout << "��ά����һ�д�СΪ��" << sizeof(arr[0]) << endl;
  cout << "��ά����Ԫ�ش�СΪ��" << sizeof(arr[0][0]) << endl;

  cout << "��ά��������Ϊ��" << sizeof(arr) / sizeof(arr[0]) << endl;
  cout << "��ά��������Ϊ��" << sizeof(arr[0]) / sizeof(arr[0][0]) << endl;

  /*
  > �ܽ�1����ά�������������������׵�ַ
  > �ܽ�2���Զ�ά����������sizeofʱ�����Ի�ȡ������ά����ռ�õ��ڴ�ռ��С
  */
  system("pause");
  return 0;
}
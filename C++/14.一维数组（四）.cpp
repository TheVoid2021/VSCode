/*
* ����Ԫ������
����������������һ��5��Ԫ�ص����飬���ҽ�Ԫ������.
(��ԭ����Ԫ��Ϊ��1,3,2,5,4;���ú�������Ϊ:4,5,2,3,1);
*/
#include <iostream>
using namespace std;

int main()
{

  int arr[5];
  cout << "�����������: " << endl;
  for (int i = 0; i < 5; i++)
  {
    cin >> arr[i];
  }
  cout << "���ú�Ľ��Ϊ��" << endl;
  for (int j = 4; j < 5 && j > -1; j--)
  {
    cout << arr[j] << " ";
  }

  system("pause");
  return 0;
}
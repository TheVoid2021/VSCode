/*
ð������
���ã� ��õ������㷨����������Ԫ�ؽ�������
1. �Ƚ����ڵ�Ԫ�ء������һ���ȵڶ����󣬾ͽ�������������
2. ��ÿһ������Ԫ����ͬ���Ĺ�����ִ����Ϻ��ҵ���һ�����ֵ��
3. �ظ����ϵĲ��裬ÿ�αȽϴ���-1��ֱ������Ҫ�Ƚ�
*/
#include <iostream>
using namespace std;

int main()
{
  int arr[9] = {4, 2, 8, 0, 5, 7, 1, 3, 9};

  for (int i = 0; i < 9 - 1; i++)
  {
    for (int j = 0; j < 9 - 1 - i; j++)
    {
      if (arr[j] > arr[j + 1])
      {
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }

  for (int i = 0; i < 9; i++)
  {
    cout << arr[i] << endl;
  }

  system("pause");

  return 0;
}

/*
* ��ֻС�������
����������
��һ�������м�¼����ֻС������أ��磺int arr[5] = {300,350,200,400,250};
�ҳ�����ӡ���ص�С�����ء�
*/
#include <iostream>
using namespace std;

int main()
{
  // int arr[5] = { 300,350,200,400,250 };
  // int maxheight = 0;
  // for (int i = 0; i < 5; i++) {
  //	if (arr[i] > maxheight) {
  //		maxheight = arr[i];
  //	}
  // }

  // cout << "���ص�С������Ϊ��" << maxheight << endl;

  int weight[5];
  int heaviest = 0;

  // ���벢������ֻС�������
  for (int i = 0; i < 5; i++)
  {
    cout << "�������" << i + 1 << "ֻС������أ�";
    cin >> weight[i];
    if (weight[i] > heaviest)
    {
      heaviest = weight[i];
    }
  }

  // ��������a�洢�м�ֻ���ص�С��
  int a = 0;

  // �ж���ֻС�����أ������
  cout << "��";
  for (int i2 = 0; i2 < 5; i2++)
  {
    if (heaviest == weight[i2])
    {
      a += 1;
      // ���ͬʱ����ֻ����С�����أ����������м�Ӷٺ�
      if (a > 1)
      {
        cout << "��";
      }
      cout << i2 + 1;
    }
  }
  cout << "ֻС�������ص�С������Ϊ" << heaviest << "��" << endl;

  system("pause");

  return 0;
}
/*break���
  ����: ��������ѡ��ṹ����ѭ���ṹ
*/
#include <iostream>
using namespace std;

int main()
{
  // 1����switch �����ʹ��break
  cout << "��ѡ������ս�������Ѷȣ�" << endl;
  cout << "1����ͨ" << endl;
  cout << "2���е�" << endl;
  cout << "3������" << endl;

  int num = 0;

  cin >> num;

  switch (num)
  {
  case 1:
    cout << "��ѡ�������ͨ�Ѷ�" << endl;
    break;
  case 2:
    cout << "��ѡ������е��Ѷ�" << endl;
    break;
  case 3:
    cout << "��ѡ����������Ѷ�" << endl;
    break;
  }

  // 2����ѭ���������break
  for (int i = 0; i < 10; i++)
  {
    if (i == 5)
    {
      break; // ����ѭ�����
    }
    cout << i << endl;
  }

  // ��Ƕ��ѭ�������ʹ��break���˳��ڲ�ѭ��
  for (int i = 0; i < 10; i++)
  {
    for (int j = 0; j < 10; j++)
    {
      if (j == 5)
      {
        break;
      }
      cout << "*" << " ";
    }
    cout << endl;
  }

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

int main()
{

  // �����Ӱ����
  // 10 ~ 9   ����
  //  8 ~ 7   �ǳ���
  //  6 ~ 5   һ��
  //  5������ ��Ƭ

  int score = 0;
  cout << "�����Ӱ���" << endl;
  cin >> score;

  switch (score)
  {
  case 10:
  case 9:
    cout << "����" << endl;
    break;
  case 8:
    cout << "�ǳ���" << endl;
    break;
  case 7:
  case 6:
    cout << "һ��" << endl;
    break;
  default: // default�����þ���switch��������е�case��������ʱ��Ҫִ�е����
    cout << "��Ƭ" << endl;
    break;
  }

  system("pause");

  return 0;
}

// switch������
//
// 1��switch���ǳ����ã�����ʹ��ʱ�����������д���κ�switch��䶼������ѭ���¹���
//
// 2��ֻ����Ի������������е���������ʹ��switch����Щ���Ͱ���int��char�ȡ������������ͣ������ʹ��if��䡣
//
// 3��switch()�Ĳ������Ͳ���Ϊʵ�� ��
//
// 4��case��ǩ�����ǳ������ʽ(constantExpression)����42����'4'��
//
// 5��case��ǩ������Ωһ�Եı��ʽ��Ҳ����˵������������case������ͬ��ֵ��
#include <iostream>
using namespace std;

int main()
{

  // �Ӽ��˳�
  int a1 = 10;
  int b1 = 3;
  /*
    cout << a1 + b1 << endl;
    cout << a1 - b1 << endl;
    cout << a1 * b1 << endl;
    cout << a1 / b1 << endl;*/
  // ����������� �����Ȼ����������С������ȥ�� ��������Ϊ0

  // 1��ǰ�õ���
  int a = 10;
  ++a;
  cout << "a=" << a << endl;

  // 2�����õ���
  int b = 10;
  b++;
  cout << "b=" << b << endl;

  // 3��ǰ�úͺ��õ�����
  // ǰ�õ��� ���ñ���+1 Ȼ����б��ʽ����
  int a2 = 10;
  int b2 = ++a2 * 10;
  cout << b2 << endl;
  cout << a2 << endl;
  // ���õ��� �Ƚ��б��ʽ���㣬���ñ���+1
  int a3 = 10;
  int b3 = a3++ * 10;

  cout << b3 << endl;
  cout << a3 << endl;

  // �߼���
  int a4 = 10;
  int a5 = 0;
  cout << (a4 || a5) << endl;

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;
// #define Day 7

int main()
{

  // cout << "HelloWorld" << endl;

  /*��ʶ����������
   * ��ʶ�������ǹؼ���
   * ��ʶ��ֻ������ĸ ���� �»������
   * ��һ���ַ�����Ϊ��ĸ���»���
   * ��ĸ���ִ�Сд
   */

  /*
   * ��д�����ݼ���
   * ��ѡ������ע�ͣ�Ctrl+K,Ctrl+C
   * ѡ������ȡ��ע�ͣ�Ctrl+K,Ctrl+U
   */

  /*1.��#define���峣��
  cout << "һ��һ����" << day << "��" << endl;
  2.��const���峣��  const���εı�����Ϊ����
  const int month = 12;
  cout << "һ�깲��" << month << "����" << endl;*/

  // 2 4 4 8
  /*
  cout << "short ������ռ�ռ�Ϊ��" << sizeof(short) << endl;

  cout << "int ������ռ�ռ�Ϊ��" << sizeof(int) << endl;

  cout << "long ������ռ�ռ�Ϊ��" << sizeof(long) << endl;

  cout << "long long ������ռ�ռ�Ϊ��" << sizeof(long long) << endl;*/

  // 4 8
  /*float f1 = 3.14f;
  double d1 = 3.14;

  cout << f1 << endl;
  cout << d1 << endl;

  cout << "float sizeof = " << sizeof(f1) << endl;
  cout << "double sizeof =" << sizeof(d1) << endl;*/

  // ��ѧ������

  float f2 = 3e2; // 3 * 10^2
  cout << "f2=" << f2 << endl;

  float f3 = 3e-2; // 3 * 0.1 ^ 2
  cout << "f3=" << f3 << endl;

  system("pause");

  return 0;
}

// VS �г��õ�һЩ��ݼ�

// һ�������Զ�����
// CTRL + K + F
//
// �������� / ������
// 1������-- - ʹ����ϼ���Ctrl + Z�����г�������
// 2��������-- - ʹ����ϼ���Ctrl + Y�����з���������
//
// ��������������ʾ
// ʹ����ϼ���Ctrl + J������ʹ����ϼ���Alt + ���������ڲ���ȫ����ؼ���ʱϵͳ�Զ�������ʾ
//
// �ġ��������ػ���ʾ��ǰ�����
// 1��ctrl + M + M
//
// ������M
//
// visual studio 2013 �г��õ�һЩ��ݼ�
//
// �塢�ص���һ�����λ�� / ǰ������һ�����λ��
// 1���ص���һ�����λ��ʹ����ϼ���Ctrl + -��
//
// 2��ǰ������һ�����λ��ʹ�á�Ctrl + Shift + -��
//
// ����ע�� / ȡ��ע��
// 1��ע������ϼ���Ctrl + K + C��
//
//  ȫע��ΪCtrl+k+c��ȡ��ע��Ctrl+k+u
//
// 2��ȡ��ע������ϼ���Ctrl + K + U��
//
// �ߡ��������
// 1�����öϵ�-- - F9
//
// 2����������-- - F5
//
// 3����������-- - F11
//
// 4������̵���-- - F10
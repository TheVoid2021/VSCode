#include <iostream>
using namespace std;

/*
! �����������������
! ������������� ()  Ҳ��������
! �������غ�ʹ�õķ�ʽ�ǳ������ĵ��ã���˳�Ϊ�º���
* �º���û�й̶�д�����ǳ����
 */

// todo ��ӡ��
class MyPrint
{
public:
  void operator()(string text)
  {
    cout << text << endl;
  }
};
void test01()
{
  // !���صģ��������� Ҳ��Ϊ�º��� ��Ϊʹ�������ǳ������ں�������
  MyPrint myFunc;
  myFunc("hello world");
}

// todo �ӷ���
class MyAdd
{
public:
  int operator()(int v1, int v2)
  {
    return v1 + v2;
  }
};

void test02()
{
  MyAdd add;
  int ret = add(10, 10); // todo �����()���ǵ�����?�ӷ������صĺ������������
  cout << "ret = " << ret << endl;

  // todo �����������    MyAdd()������һ����������  ����Ҫ��?MyAdd add
  cout << "MyAdd()(100,100) = " << MyAdd()(100, 100) << endl;
}

int main()
{

  test01();
  test02();

  system("pause");

  return 0;
}
#include <iostream>
using namespace std;

// !��ָ����ʳ�Ա����
class Person
{
public:
  void ShowClassName()
  {
    cout << "����Person��!" << endl;
  }

  void ShowPerson()
  {
    // cout << "����Ϊ��" << mAge << endl; // �ȼ���this->mAge��Ĭ�ϵģ�thisָ��ǰ�������洫����ǿ�ָ�룬���Իᱨ��
    // !����õ�thisָ�룬��Ҫ�ȼ����жϱ�֤����Ľ�׳��
    if (this == NULL)
    {
      return;
    }
    cout << mAge << endl;
  }

public:
  int mAge;
};

void test01()
{
  Person *p = NULL;
  p->ShowClassName(); // !��ָ�룬���Ե��ó�Ա����
  p->ShowPerson();    // ���������Ա�������õ���thisָ�룬�Ͳ������ˣ���Ҫ�����ж�
}

int main()
{

  test01();

  system("pause");

  return 0;
}
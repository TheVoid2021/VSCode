#include <iostream>
using namespace std;

/*
��������ļ��п���(Public)�����������(Private)
�����������Ŀ��˶����Խ�ȥ���������������˽�еģ�Ҳ����˵ֻ�����ܽ�ȥ
�����أ���Ҳ����������ĺù��ۺû��ѽ�ȥ��

! �ڳ������Щ˽������ Ҳ�������������һЩ������������з��ʣ�����Ҫ�õ���Ԫ�ļ���
! ��Ԫ��Ŀ�ľ�����һ������������ ������һ������˽�г�Ա
! ��Ԫ�Ĺؼ���Ϊ  friend
?��Ԫ������ʵ��
  * ȫ�ֺ�������Ԫ
  * ������Ԫ
  * ��Ա��������Ԫ
 */

// todo ȫ�ֺ�������Ԫ
class Building
{
  // todo ���߱����� goodGayȫ�ֺ��� �� Building��ĺ����ѣ����Է������е�˽������
  friend void goodGay(Building *building); // todo friend + ��������

public:
  Building()
  {
    this->m_SittingRoom = "����";
    this->m_BedRoom = "����";
  }

public:
  string m_SittingRoom; // ����

private:
  string m_BedRoom; // ����
};

void goodGay(Building *building)
{
  cout << "�û������ڷ��ʣ� " << building->m_SittingRoom << endl;
  cout << "�û������ڷ��ʣ� " << building->m_BedRoom << endl;
}

void test01()
{
  Building b;
  goodGay(&b);
}

int main()
{

  test01();

  system("pause");
  return 0;
}
#include <iostream>
using namespace std;

// !��Ա��������Ԫ
class Building;
class goodGay
{
public:
  goodGay();
  void visit(); // !ֻ��visit������ΪBuilding�ĺ����ѣ����Է�����Building��˽������
  void visit2();

private:
  Building *building;
};

class Building
{
  // !���߱�����  goodGay���е�visit��Ա���� ��Building�����ѣ����Է���˽������
  friend void goodGay::visit();

public:
  Building();

public:
  string m_SittingRoom; // ����
private:
  string m_BedRoom; // ����
};

// todo ����ʵ�ֳ�Ա����
Building::Building()
{
  this->m_SittingRoom = "����";
  this->m_BedRoom = "����";
}

// todo ����ʵ�ֳ�Ա����
goodGay::goodGay()
{
  building = new Building;
}

// todo ����ʵ�ֳ�Ա����
void goodGay::visit()
{
  cout << "�û������ڷ���" << building->m_SittingRoom << endl;
  cout << "�û������ڷ���" << building->m_BedRoom << endl; // todo ������m_BedRoom�ķ���Ȩ�޸�visit
}

void goodGay::visit2()
{
  cout << "�û������ڷ���" << building->m_SittingRoom << endl;
  // cout << "�û������ڷ���" << building->m_BedRoom << endl;  // todo ֻ��visit��Ա������������ԪȨ��
}

void test01()
{
  goodGay gg;
  gg.visit();
}

int main()
{

  test01();

  system("pause");
  return 0;
}
#include <iostream>
using namespace std;

/*
! ���μ̳и��
  * ����������̳�ͬһ������
  * ����ĳ����ͬʱ�̳�������������
  * ���ּ̳б���Ϊ���μ̳У�������ʯ�̳�
* ���μ̳д�������Ҫ����������̳�������ͬ�����ݣ�������Դ�˷��Լ���������
* ������̳п��Խ�����μ̳�����
 */

class Animal
{
public:
  int m_Age;
};

// ! �̳�ǰ��virtual�ؼ��ֺ󣬱�Ϊ��̳�
// ! ��ʱ�����ĸ���Animal��Ϊ�����
class Sheep : virtual public Animal
{
};
class Tuo : virtual public Animal
{
};
class SheepTuo : public Sheep, public Tuo
{
};

void test01()
{
  SheepTuo st;
  st.Sheep::m_Age = 100;
  st.Tuo::m_Age = 200;

  // todo �����μ̳�ʱ����������ӵ����ͬ���ݣ���Ҫ��������������
  cout << "st.Sheep::m_Age = " << st.Sheep::m_Age << endl;
  cout << "st.Tuo::m_Age = " << st.Tuo::m_Age << endl;
  cout << "st.m_Age = " << st.m_Age << endl; // todo ��̳к�����̳е�����ֻ��һ�������Բ�������������
  // ! ��̳���ʵ�������������һ��ָ�룬ָ������࣬ͨ��ָ���������������
}

int main()
{

  test01();

  system("pause");

  return 0;
}
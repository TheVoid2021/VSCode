#include <iostream>
using namespace std;

/*
! �������������
! ���ã�ͨ�����ص����������ʵ���Զ���ĵ������㣬ʵ���Լ�����������
! ǰ�õ����������ã����õ�������ֵ
 */

class MyInteger
{
  friend ostream &operator<<(ostream &out, MyInteger myint);

public:
  MyInteger()
  {
    m_Num = 0;
  }
  // todo ����ǰ��++�����
  MyInteger &operator++() // !����������Ϊ��ʵ�������������ñ�����ŵ���������ͬһ�����ݽ��е�����������ÿ�ε���������һ���µĶ���
  {
    // todo �Ƚ���++����
    m_Num++;
    // todo �ٽ�������з���
    return *this;
  }

  // todo ���غ���++�����
  MyInteger operator++(int) // ?int����ռλ������ֻ��Ϊ������ǰ�úͺ��ã�û��ʵ������
  {
    // todo �ȼ�¼��ǰ���
    MyInteger temp = *this; // ?��¼��ǰ�����ֵ��Ȼ���ñ����ֵ��1�����Ƿ��ص�����ǰ��ֵ���ﵽ�ȷ��غ�++��
    // todo ����
    m_Num++;
    // ! ��󷵻ؼ�¼��ֵ ���������ã�����Ϊtemp��һ���ֲ�����������ִ����ϣ�temp�ͻᱻ�ͷţ��������þͻᱨ��
    return temp;
  }

private:
  int m_Num;
};

// *����<<�����
ostream &operator<<(ostream &out, MyInteger myint)
{
  out << myint.m_Num;
  return out;
}

// ǰ��++ ��++ �ٷ���
void test01()
{
  MyInteger myInt;
  cout << ++myInt << endl;
  cout << myInt << endl;
}

// ����++ �ȷ��� ��++
void test02()
{

  MyInteger myInt;
  cout << myInt++ << endl;
  cout << myInt << endl;
}

int main()
{

  test01();
  test02();

  system("pause");

  return 0;
}
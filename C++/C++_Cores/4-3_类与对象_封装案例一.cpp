#include <iostream>
using namespace std;

/*
!�����������
*1��������������
*2���������
*3�������Ϊ ��ȡ��������������
*4���ֱ�����ȫ�ֺ����ͳ�Ա���� �ж������������Ƿ���� */

class Cube
{
public:
  // ���ó���
  void setL(int l)
  {

    m_L = l;
  }
  // ���ÿ��
  void setW(int w)
  {

    m_W = w;
  }
  // ���ø߶�
  void setH(int h)
  {

    m_H = h;
  }
  // ��ȡ����
  int getL()
  {
    return m_L;
  }
  // ��ȡ���
  int getW()
  {
    return m_W;
  }
  // ��ȡ�߶�
  int getH()
  {
    return m_H;
  }
  // ��ȡ���
  int getArea()
  {
    return 2 * (m_L * m_W + m_L * m_H + m_W * m_H);
  }
  // ��ȡ���
  int getVolume()
  {
    return m_L * m_W * m_H;
  }

  // !���ó�Ա�����ж������������Ƿ���� ֻ��Ҫ��һ������
  bool isSameByClass(Cube &c2)
  {
    if (m_L == c2.getL() && m_W == c2.getW() && m_H == c2.getH())
    {
      return true;
    }
    else
    {
      return false;
    }
  }

private:
  int m_L; // ����
  int m_W; // ���
  int m_H; // �߶�
};

// !����ȫ�ֺ����ж������������Ƿ���� Ҫ����������
bool isSame(Cube &c1, Cube &c2)
{
  if (c1.getL() == c2.getL() && c1.getW() == c2.getW() && c1.getH() == c2.getH())
  {
    return true;
  }
  else
  {
    return false;
  }
}

int main()
{
  // ������һ��������
  Cube c1;
  c1.setL(10);
  c1.setW(10);
  c1.setH(10);
  cout << "�����" << c1.getArea() << endl;
  cout << "�����" << c1.getVolume() << endl;

  // �����ڶ���������
  Cube c2;
  c2.setL(10);
  c2.setW(10);
  c2.setH(10);

  // todo ����ȫ�ֺ����ж������������Ƿ����
  if (isSame(c1, c2))
  {
    cout << "�������������" << endl;
  }
  else
  {
    cout << "���������岻���" << endl;
  }

  // todo ���ó�Ա�����ж������������Ƿ����
  if (c1.isSameByClass(c2))
  {
    cout << "�������������" << endl;
  }
  else
  {
    cout << "���������岻���" << endl;
  }
  return 0;
}
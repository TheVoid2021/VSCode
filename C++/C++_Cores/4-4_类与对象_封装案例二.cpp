#include <iostream>
using namespace std;

/*
!���Բ��ϵ����
!�����п�������һ������Ϊ�����еĳ�Ա
!��ΰѶ�����ɶ���ļ���д ����VSʵ��
 */

// ����
class Point
{
public:
  // ����x
  void setX(int x)
  {
    m_X = x;
  }
  // ��ȡx
  int getX()
  {
    return m_X;
  }
  // ����y
  void setY(int y)
  {
    m_Y = y;
  }
  // ��ȡy
  int getY()
  {
    return m_Y;
  }

private:
  int m_X;
  int m_Y;
};

// Բ��
class Circle
{
public:
  // ����Բ��
  void setCenter(Point center)
  {
    m_Center = center;
  }
  // ��ȡԲ��
  Point getCenter()
  {
    return m_Center;
  }
  // ���ð뾶
  void setRadius(int radius)
  {
    m_Radius = radius;
  }
  // ��ȡ�뾶
  int getRadius()
  {
    return m_Radius;
  }

private:
  int m_Radius; // �뾶
  // !�����п�������һ������Ϊ�����еĳ�Ա
  Point m_Center; // Բ��
};

// �жϵ��Բ�Ĺ�ϵ
void relation(Circle &circle, Point &point)
{
  // �������Բ�ĵľ��� ƽ��
  int distance = (circle.getCenter().getX() - point.getX()) * (circle.getCenter().getX() - point.getX()) +
                 (circle.getCenter().getY() - point.getY()) * (circle.getCenter().getY() - point.getY());
  // ����뾶��ƽ��
  int radius = circle.getRadius() * circle.getRadius();
  // �жϹ�ϵ
  if (distance > radius)
  {
    cout << "����Բ��" << endl;
  }
  else if (distance == radius)
  {
    cout << "����Բ��" << endl;
  }
  else
  {
    cout << "����Բ��" << endl;
  }
}

int main()
{
  // ����Բ����
  Circle circle;
  // ����Բ��
  Point center;
  center.setX(10);
  center.setY(0);
  circle.setCenter(center);
  // ���ð뾶
  circle.setRadius(10);

  // ���������
  Point point;
  point.setX(12);
  point.setY(10);

  // �жϵ��Բ�Ĺ�ϵ
  relation(circle, point);
}
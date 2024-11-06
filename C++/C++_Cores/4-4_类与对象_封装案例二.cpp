#include <iostream>
using namespace std;

/*
!点和圆关系案例
!在类中可以让另一个类作为本类中的成员
!如何把多个类拆成多个文件编写 可用VS实现
 */

// 点类
class Point
{
public:
  // 设置x
  void setX(int x)
  {
    m_X = x;
  }
  // 获取x
  int getX()
  {
    return m_X;
  }
  // 设置y
  void setY(int y)
  {
    m_Y = y;
  }
  // 获取y
  int getY()
  {
    return m_Y;
  }

private:
  int m_X;
  int m_Y;
};

// 圆类
class Circle
{
public:
  // 设置圆心
  void setCenter(Point center)
  {
    m_Center = center;
  }
  // 获取圆心
  Point getCenter()
  {
    return m_Center;
  }
  // 设置半径
  void setRadius(int radius)
  {
    m_Radius = radius;
  }
  // 获取半径
  int getRadius()
  {
    return m_Radius;
  }

private:
  int m_Radius; // 半径
  // !在类中可以让另一个类作为本类中的成员
  Point m_Center; // 圆心
};

// 判断点和圆的关系
void relation(Circle &circle, Point &point)
{
  // 计算点与圆心的距离 平方
  int distance = (circle.getCenter().getX() - point.getX()) * (circle.getCenter().getX() - point.getX()) +
                 (circle.getCenter().getY() - point.getY()) * (circle.getCenter().getY() - point.getY());
  // 计算半径的平方
  int radius = circle.getRadius() * circle.getRadius();
  // 判断关系
  if (distance > radius)
  {
    cout << "点在圆外" << endl;
  }
  else if (distance == radius)
  {
    cout << "点在圆上" << endl;
  }
  else
  {
    cout << "点在圆内" << endl;
  }
}

int main()
{
  // 创建圆对象
  Circle circle;
  // 设置圆心
  Point center;
  center.setX(10);
  center.setY(0);
  circle.setCenter(center);
  // 设置半径
  circle.setRadius(10);

  // 创建点对象
  Point point;
  point.setX(12);
  point.setY(10);

  // 判断点和圆的关系
  relation(circle, point);
}
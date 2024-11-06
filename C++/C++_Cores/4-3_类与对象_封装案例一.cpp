#include <iostream>
using namespace std;

/*
!立方体类设计
*1、创建立方体类
*2、设计属性
*3、设计行为 获取立方体面积和体积
*4、分别利用全局函数和成员函数 判断两个立方体是否相等 */

class Cube
{
public:
  // 设置长度
  void setL(int l)
  {

    m_L = l;
  }
  // 设置宽度
  void setW(int w)
  {

    m_W = w;
  }
  // 设置高度
  void setH(int h)
  {

    m_H = h;
  }
  // 获取长度
  int getL()
  {
    return m_L;
  }
  // 获取宽度
  int getW()
  {
    return m_W;
  }
  // 获取高度
  int getH()
  {
    return m_H;
  }
  // 获取面积
  int getArea()
  {
    return 2 * (m_L * m_W + m_L * m_H + m_W * m_H);
  }
  // 获取体积
  int getVolume()
  {
    return m_L * m_W * m_H;
  }

  // !利用成员函数判断两个立方体是否相等 只需要传一个参数
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
  int m_L; // 长度
  int m_W; // 宽度
  int m_H; // 高度
};

// !利用全局函数判断两个立方体是否相等 要传两个参数
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
  // 创建第一个立方体
  Cube c1;
  c1.setL(10);
  c1.setW(10);
  c1.setH(10);
  cout << "面积：" << c1.getArea() << endl;
  cout << "体积：" << c1.getVolume() << endl;

  // 创建第二个立方体
  Cube c2;
  c2.setL(10);
  c2.setW(10);
  c2.setH(10);

  // todo 利用全局函数判断两个立方体是否相等
  if (isSame(c1, c2))
  {
    cout << "两个立方体相等" << endl;
  }
  else
  {
    cout << "两个立方体不相等" << endl;
  }

  // todo 利用成员函数判断两个立方体是否相等
  if (c1.isSameByClass(c2))
  {
    cout << "两个立方体相等" << endl;
  }
  else
  {
    cout << "两个立方体不相等" << endl;
  }
  return 0;
}
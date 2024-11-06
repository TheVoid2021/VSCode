#include <iostream>
using namespace std;

/*
!C++ 面向对象的三大特性为： == 封装、继承、多态 ==

    *C++ 认为 == 万事万物都皆为对象 ==，对象上有其属性和行为

**例如： **
? 人可以作为对象，属性有姓名、年龄、身高、体重...，行为有走、跑、跳、吃饭、唱歌...
? 车也可以作为对象，属性有轮胎、方向盘、车灯...,行为有载人、放音乐、放空调...
? 具有相同性质的 == 对象 ==，我们可以抽象称为 == 类 ==，人属于人类，车属于车类 */

/*
!封装是C++ 面向对象三大特性之一

  *封装的意义：

    *将属性和行为作为一个整体，表现生活中的事物
    *将属性和行为加以权限控制

      *封装意义一：
        在设计类的时候，属性和行为写在一起，表现事物
  *语法：
todo class 类名
todo  {
todo  访问权限： 属性 / 行为
todo  };

? 类中的属性和行为统称为成员，封装之后的成员都称为类的成员
? 属性 成员属性 / 成员变量
? 行为 成员函数 / 成员方法
*/

// 示例1：设计一个圆类，求圆的周长
// 圆周率
const double PI = 3.14;

// 1、封装的意义
// 将属性和行为作为一个整体，用来表现生活中的事物

// 封装一个圆类，求圆的周长
// !class代表设计一个类，后面跟着的是类名
class Circle
{
public: // *访问权限  公共的权限
  // !属性
  int m_r; // 半径

  // !行为
  // 获取到圆的周长
  double calculateZC()
  {
    // 2 * pi  * r
    // 获取圆的周长
    return 2 * PI * m_r;
  }
};

// 示例2：设计一个学生类，属性有姓名和学号，可以给姓名和学号赋值，可以显示学生的姓名和学号
class Student
{
public:
  void setName(string name) // 通过函数设置姓名
  {
    m_Name = name;
  }
  void setId(int id) // 通过函数设置学号
  {
    m_Id = id;
  }
  string m_Name;
  int m_Id;

  void showStudent()
  {
    cout << "姓名：" << m_Name << " 学号：" << m_Id << endl;
  }
};

int main()
{

  // !通过圆类，创建圆的对象的过程，称为实例化
  //  c1就是一个具体的圆
  Circle c1;
  c1.m_r = 10; // 给圆对象的半径 进行赋值操作
  // 2 * pi * 10 = = 62.8
  cout << "圆的周长为： " << c1.calculateZC() << endl;

  Student stu;
  stu.setName("张三");
  stu.setId(10001);
  stu.showStudent();

  system("pause");

  return 0;
}
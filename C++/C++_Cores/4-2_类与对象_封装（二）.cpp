#include <iostream>
using namespace std;

/*
!封装意义二：权限访问设置
*类在设计时，可以把属性和行为放在不同的权限下，加以控制

*访问权限有三种：
todo 1. public       公共权限    类内可以访问，类外可以访问
todo 2. protected    保护权限    类内可以访问，类外不可以访问，子类可以访问
todo 3. private      私有权限    类内可以访问，类外不可以访问，子类不可以访问
*/
class Person
{
public:
  // 公共权限
  string m_Name; // 姓名

protected:
  // 保护权限
  string m_Car; // 车子

private:
  // 私有权限
  int m_Password; // 银行卡密码

public:
  void func()
  {
    m_Name = "张三";
    m_Car = "拖拉机";
    m_Password = 123456;
  }
};

/*
!在C++中 struct和class唯一的区别就在于 默认的访问权限不同
*区别：
todo struct  默认权限为公共
todo class   默认权限为私有 */
class c1
{
  int m_A; // 默认是私有权限
};

struct s1
{
  int m_A; // 默认是公共权限
};

/*
!成员属性设置为私有
*优点1：将所有成员属性设置为私有，可以自己控制读写权限
*优点2：对于写权限，我们可以检测数据的有效性 */
class Person1
{
public:
  // 姓名设置可读可写
  void setName(string name) // 写
  {
    m_name = name;
  }
  string getName() // 读
  {
    return m_name;
  }

  // 获取年龄
  int getAge()
  {
    return m_Age;
  }
  // 设置年龄
  void setAge(int age)
  {
    if (age < 0 || age > 150)
    {
      cout << "你个老妖精!" << endl;
      return;
    }
    m_Age = age;
  }

  // 情人设置为只写
  void setLover(string lover)
  {
    m_Lover = lover;
  }

private:
  string m_name; // 可读可写  姓名

  int m_Age; // 只读  年龄

  string m_Lover; // 只写  情人
};

//! 主程序
int main()
{
  Person p1;
  p1.m_Name = "李四";
  // p1.m_Car = "奔驰"; // 错误，保护权限，类外不可以访问
  // p1.m_Password = 123456; // 错误，私有权限，类外不可以访问
  p1.func();

  // c1 c;
  //  c.m_A = 100; // 错误，默认是私有权限，类外不可以访问

  // s1 s;
  // s.m_A = 100; // 正确，默认是公共权限，类外可以访问

  Person1 p;
  // 姓名设置
  p.setName("张三");
  cout << "姓名： " << p.getName() << endl;

  // 年龄设置
  p.setAge(50); // todo写权限，检测数据的有效性
  // p.m_Age = 100; // 错误，只读权限，类外不可以访问
  cout << "年龄： " << p.getAge() << endl;

  // 情人设置
  p.setLover("苍井");
  // cout << "情人： " << p.m_Lover << endl;  //只写属性，不可以读取
  return 0;
}
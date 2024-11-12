#include <iostream>
using namespace std;

/*
! 职工管理系统可以用来管理公司内所有员工的信息
* 本教程主要利用C++来实现一个基于多态的职工管理系统
? 公司中职工分为三类：普通员工、经理、老板，显示信息时，需要显示职工编号、职工姓名、职工岗位、以及职责
    * 普通员工职责：完成经理交给的任务
    * 经理职责：完成老板交给的任务，并下发任务给员工
    * 老板职责：管理公司所有事务
 */

/*
! 职工抽象基类
  * 职工的分类为：普通员工、经理、老板
  * 将三种职工抽象到一个类（worker）中,利用多态管理不同职工种类
  * 职工的属性为：职工编号、职工姓名、职工所在部门编号
  * 职工的行为为：岗位职责信息描述，获取岗位名称*/
class Worker
{
public:
  virtual ~Worker() = default; // 添加虚析构函数，解决释放子类对象时，无法调用子类析构函数的问题
  // 显示个人信息
  virtual void showInfo() = 0;
  // 获取岗位名称
  virtual string getDeptName() = 0;

  int m_Id;      // 职工编号
  string m_Name; // 职工姓名
  int m_DeptId;  // 职工所在部门名称编号
};

/*
! 普通员工类
* 普通员工类 继承 职工抽象类，并重写父类中纯虚函数
 */
class Employee : public Worker
{
public:
  // 构造函数
  Employee(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // 显示个人信息
  virtual void showInfo()
  {
    cout << "职工编号： " << this->m_Id
         << " \t职工姓名： " << this->m_Name
         << " \t岗位：" << this->getDeptName()
         << " \t岗位职责：完成经理交给的任务" << endl;
  }

  // 获取职工岗位名称
  virtual string getDeptName()
  {
    return string("员工");
  }
};

/*
! 经理类
* 经理类 继承 职工抽象类，并重写父类中纯虚函数，和普通员工类似 */
class Manager : public Worker
{
public:
  Manager(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // 显示个人信息
  virtual void showInfo()
  {
    cout << "职工编号： " << this->m_Id
         << " \t职工姓名： " << this->m_Name
         << " \t岗位：" << this->getDeptName()
         << " \t岗位职责：完成老板交给的任务,并下发任务给员工" << endl;
  }

  // 获取职工岗位名称
  virtual string getDeptName()
  {
    return string("经理");
  }
};

/*
! 老板类
* 老板类 继承 职工抽象类，并重写父类中纯虚函数，和普通员工类似 */
class Boss : public Worker
{
public:
  Boss(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // 显示个人信息
  virtual void showInfo()
  {
    cout << "职工编号： " << this->m_Id
         << " \t职工姓名： " << this->m_Name
         << " \t岗位：" << this->getDeptName()
         << " \t岗位职责：管理公司所有事务" << endl;
  }

  // 获取职工岗位名称
  virtual string getDeptName()
  {
    return string("总裁");
  }
};

// ! 工人管理者类
class WorkerManager
{
public:
  // 构造函数
  WorkerManager()
  {
    // 初始化人数
    this->m_EmpNum = 0;

    // 初始化数组指针
    this->m_EmpArray = NULL;
  }
  void Show_Menu()
  {
    cout << "********************************************" << endl;
    cout << "*********  欢迎使用职工管理系统！ **********" << endl;
    cout << "*************  0.退出管理程序  *************" << endl;
    cout << "*************  1.增加职工信息  *************" << endl;
    cout << "*************  2.显示职工信息  *************" << endl;
    cout << "*************  3.删除离职职工  *************" << endl;
    cout << "*************  4.修改职工信息  *************" << endl;
    cout << "*************  5.查找职工信息  *************" << endl;
    cout << "*************  6.按照编号排序  *************" << endl;
    cout << "*************  7.清空所有文档  *************" << endl;
    cout << "********************************************" << endl;
    cout << endl;
  }

  // 退出系统
  void exitSystem()
  {
    cout << "欢迎下次使用" << endl;
    system("pause");
    exit(0);
  }

  // 添加职工
  void Add_Emp()
  {
    cout << "请输入增加职工数量： " << endl;

    int addNum = 0;
    cin >> addNum;

    if (addNum > 0)
    {
      // 计算新空间大小
      int newSize = this->m_EmpNum + addNum;

      // 开辟新空间
      Worker **newSpace = new Worker *[newSize];

      // 将原空间下内容存放到新空间下
      if (this->m_EmpArray != NULL)
      {
        for (int i = 0; i < this->m_EmpNum; i++)
        {
          newSpace[i] = this->m_EmpArray[i];
        }
      }

      // 输入新数据
      for (int i = 0; i < addNum; i++)
      {
        int id;
        string name;
        int dSelect;

        cout << "请输入第 " << i + 1 << " 个新职工编号：" << endl;
        cin >> id;

        cout << "请输入第 " << i + 1 << " 个新职工姓名：" << endl;
        cin >> name;

        cout << "请选择该职工的岗位：" << endl;
        cout << "1、普通职工" << endl;
        cout << "2、经理" << endl;
        cout << "3、老板" << endl;
        cin >> dSelect;

        Worker *worker = NULL;
        switch (dSelect)
        {
        case 1: // 普通员工
          worker = new Employee(id, name, 1);
          break;
        case 2: // 经理
          worker = new Manager(id, name, 2);
          break;
        case 3: // 老板
          worker = new Boss(id, name, 3);
          break;
        default:
          break;
        }

        newSpace[this->m_EmpNum + i] = worker;
      }

      // 释放原有空间
      delete[] this->m_EmpArray;

      // 更改新空间的指向
      this->m_EmpArray = newSpace;

      // 更新新的个数
      this->m_EmpNum = newSize;

      // 提示信息
      cout << "成功添加" << addNum << "名新职工！" << endl;
    }
    else
    {
      cout << "输入有误" << endl;
    }

    system("pause");
    system("cls");
  }

  // 析构函数
  ~WorkerManager()
  {
    if (this->m_EmpArray != NULL)
    {
      delete[] this->m_EmpArray;
    }
  }

public:
  // 记录文件中的人数个数
  int m_EmpNum;

  // 员工数组的指针
  Worker **m_EmpArray;
};

void test()
{
  Worker *worker = NULL;
  worker = new Employee(1, "张三", 1);
  worker->showInfo();
  delete worker;

  worker = new Manager(2, "李四", 2);
  worker->showInfo();
  delete worker;

  worker = new Boss(3, "王五", 3);
  worker->showInfo();
  delete worker;
}

int main()
{

  WorkerManager wm;

  int choice = 0;
  while (true)
  {
    // 展示菜单
    wm.Show_Menu();
    cout << "请输入您的选择:" << endl;
    cin >> choice;

    switch (choice)
    {
    case 0: // 退出系统
      wm.exitSystem();
      break;
    case 1: // 添加职工
      wm.Add_Emp();
      break;
    case 2: // 显示职工
      break;
    case 3: // 删除职工
      break;
    case 4: // 修改职工
      break;
    case 5: // 查找职工
      break;
    case 6: // 排序职工
      break;
    case 7: // 清空文件
      break;
    default:
      system("cls");
      break;
    }
  }

  // test();

  system("pause");
  return 0;
}
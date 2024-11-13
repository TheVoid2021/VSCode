#include <iostream>
#include <fstream>
#define FILENAME "empFile.txt" // 文件名
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

    ifstream ifs;
    ifs.open(FILENAME, ios::in);

    // 文件不存在情况
    if (!ifs.is_open())
    {
      cout << "文件不存在" << endl; // 测试输出
      this->m_EmpNum = 0;           // 初始化人数
      this->m_FileIsEmpty = true;   // 初始化文件为空标志
      this->m_EmpArray = NULL;      // 初始化数组
      ifs.close();                  // 关闭文件
      return;
    }
    // 文件存在，并且没有记录
    char ch;
    ifs >> ch;
    if (ifs.eof())
    {
      cout << "文件为空!" << endl;
      this->m_EmpNum = 0;
      this->m_FileIsEmpty = true;
      this->m_EmpArray = NULL;
      ifs.close();
      return;
    }
    int num = this->get_EmpNum();
    // cout << "职工个数为：" << num << endl; // 测试代码
    this->m_EmpNum = num; // 更新成员属性

    // 根据职工数创建数组
    this->m_EmpArray = new Worker *[this->m_EmpNum];
    // 初始化职工
    init_Emp();

    // 测试代码
    // for (int i = 0; i < m_EmpNum; i++)
    // {
    //   cout << "职工号： " << this->m_EmpArray[i]->m_Id
    //        << " 职工姓名： " << this->m_EmpArray[i]->m_Name
    //        << " 部门编号： " << this->m_EmpArray[i]->m_DeptId << endl;
    // }
  }

  void save()
  {
    ofstream ofs;
    ofs.open(FILENAME, ios::out);

    for (int i = 0; i < this->m_EmpNum; i++)
    {
      ofs << this->m_EmpArray[i]->m_Id << " "
          << this->m_EmpArray[i]->m_Name << " "
          << this->m_EmpArray[i]->m_DeptId << endl;
    }

    ofs.close();
  }

  int get_EmpNum()
  {
    ifstream ifs;
    ifs.open(FILENAME, ios::in);

    int id;
    string name;
    int dId;

    int num = 0;

    while (ifs >> id && ifs >> name && ifs >> dId)
    {
      // 记录人数
      num++;
    }
    ifs.close();

    return num;
  }

  void init_Emp()
  {
    ifstream ifs;
    ifs.open(FILENAME, ios::in);

    int id;
    string name;
    int dId;

    int index = 0;
    while (ifs >> id && ifs >> name && ifs >> dId)
    {
      Worker *worker = NULL;
      // 根据不同的部门Id创建不同对象
      if (dId == 1) // 1普通员工
      {
        worker = new Employee(id, name, dId);
      }
      else if (dId == 2) // 2经理
      {
        worker = new Manager(id, name, dId);
      }
      else // 总裁
      {
        worker = new Boss(id, name, dId);
      }
      // 存放在数组中
      this->m_EmpArray[index] = worker;
      index++;
    }
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

      // 更新职工不为空标志
      this->m_FileIsEmpty = false;

      // 提示信息
      cout << "成功添加" << addNum << "名新职工！" << endl;
    }
    else
    {
      cout << "输入有误" << endl;
    }

    // 保存到文件中
    this->save();

    system("pause");
    system("cls");
  }

  // 显示职工
  void Show_Emp()
  {
    if (this->m_FileIsEmpty)
    {
      cout << "文件不存在或记录为空！" << endl;
    }
    else
    {
      for (int i = 0; i < m_EmpNum; i++)
      {
        // 利用多态调用接口
        this->m_EmpArray[i]->showInfo();
      }
    }

    system("pause");
    system("cls");
  }

  // 删除职工之前，先检查职工是否存在
  int IsExist(int id)
  {
    int index = -1;

    for (int i = 0; i < this->m_EmpNum; i++)
    {
      if (this->m_EmpArray[i]->m_Id == id)
      {
        index = i;

        break;
      }
    }

    return index;
  }

  // 删除职工
  void Del_Emp()
  {
    if (this->m_FileIsEmpty)
    {
      cout << "文件不存在或记录为空！" << endl;
    }
    else
    {
      // 按职工编号删除
      cout << "请输入想要删除的职工号：" << endl;
      int id = 0;
      cin >> id;

      int index = this->IsExist(id);

      if (index != -1) // 说明index上位置数据需要删除
      {
        for (int i = index; i < this->m_EmpNum - 1; i++)
        {
          this->m_EmpArray[i] = this->m_EmpArray[i + 1];
        }
        this->m_EmpNum--;

        this->save(); // 删除后数据同步到文件中
        cout << "删除成功！" << endl;
      }
      else
      {
        cout << "删除失败，未找到该职工" << endl;
      }
    }

    system("pause");
    system("cls");
  }

  // 修改职工
  void Mod_Emp()
  {
    if (this->m_FileIsEmpty)
    {
      cout << "文件不存在或记录为空！" << endl;
    }
    else
    {
      cout << "请输入修改职工的编号：" << endl;
      int id;
      cin >> id;

      int ret = this->IsExist(id);
      if (ret != -1)
      {
        // 查找到编号的职工

        delete this->m_EmpArray[ret];

        int newId = 0;
        string newName = "";
        int dSelect = 0;

        cout << "查到： " << id << "号职工，请输入新职工号： " << endl;
        cin >> newId;

        cout << "请输入新姓名： " << endl;
        cin >> newName;

        cout << "请输入岗位： " << endl;
        cout << "1、普通职工" << endl;
        cout << "2、经理" << endl;
        cout << "3、老板" << endl;
        cin >> dSelect;

        Worker *worker = NULL;
        switch (dSelect)
        {
        case 1:
          worker = new Employee(newId, newName, dSelect);
          break;
        case 2:
          worker = new Manager(newId, newName, dSelect);
          break;
        case 3:
          worker = new Boss(newId, newName, dSelect);
          break;
        default:
          cout << "输入有误" << endl;
          break;
        }

        // 更改数据 到数组中
        this->m_EmpArray[ret] = worker;

        cout << "修改成功！" << endl;

        // 保存到文件中
        this->save();
      }
      else
      {
        cout << "修改失败，查无此人" << endl;
      }
    }

    // 按任意键 清屏
    system("pause");
    system("cls");
  }

  // 查找职工
  void Find_Emp()
  {
    if (this->m_FileIsEmpty)
    {
      cout << "文件不存在或记录为空！" << endl;
    }
    else
    {
      cout << "请输入查找的方式：" << endl;
      cout << "1、按职工编号查找" << endl;
      cout << "2、按姓名查找" << endl;

      int select = 0;
      cin >> select;

      if (select == 1) // 按职工号查找
      {
        int id;
        cout << "请输入查找的职工编号：" << endl;
        cin >> id;

        int ret = IsExist(id);
        if (ret != -1)
        {
          cout << "查找成功！该职工信息如下：" << endl;
          this->m_EmpArray[ret]->showInfo();
        }
        else
        {
          cout << "查找失败，查无此人" << endl;
        }
      }
      else if (select == 2) // 按姓名查找
      {
        string name;
        cout << "请输入查找的姓名：" << endl;
        cin >> name;

        bool flag = false; // 查找到的标志
        for (int i = 0; i < m_EmpNum; i++)
        {
          if (m_EmpArray[i]->m_Name == name)
          {
            cout << "查找成功,职工编号为："
                 << m_EmpArray[i]->m_Id
                 << " 号的信息如下：" << endl;

            flag = true;

            this->m_EmpArray[i]->showInfo();
          }
        }
        if (flag == false)
        {
          // 查无此人
          cout << "查找失败，查无此人" << endl;
        }
      }
      else
      {
        cout << "输入选项有误" << endl;
      }
    }

    system("pause");
    system("cls");
  }

  // 排序职工
  void Sort_Emp()
  {
    if (this->m_FileIsEmpty)
    {
      cout << "文件不存在或记录为空！" << endl;
      system("pause");
      system("cls");
    }
    else
    {
      cout << "请选择排序方式： " << endl;
      cout << "1、按职工号进行升序" << endl;
      cout << "2、按职工号进行降序" << endl;

      int select = 0;
      cin >> select;

      for (int i = 0; i < m_EmpNum; i++)
      {
        int minOrMax = i;
        for (int j = i + 1; j < m_EmpNum; j++)
        {
          if (select == 1) // 升序
          {
            if (m_EmpArray[minOrMax]->m_Id > m_EmpArray[j]->m_Id)
            {
              minOrMax = j;
            }
          }
          else // 降序
          {
            if (m_EmpArray[minOrMax]->m_Id < m_EmpArray[j]->m_Id)
            {
              minOrMax = j;
            }
          }
        }

        if (i != minOrMax)
        {
          Worker *temp = m_EmpArray[i];
          m_EmpArray[i] = m_EmpArray[minOrMax];
          m_EmpArray[minOrMax] = temp;
        }
      }

      cout << "排序成功,排序后结果为：" << endl;
      this->save();
      this->Show_Emp();
    }
  }

  // 清空文件
  void Clean_File()
  {
    cout << "确认清空？" << endl;
    cout << "1、确认" << endl;
    cout << "2、返回" << endl;

    int select = 0;
    cin >> select;

    if (select == 1)
    {
      // 打开模式 ios::trunc 如果存在删除文件并重新创建
      ofstream ofs(FILENAME, ios::trunc);
      ofs.close();

      if (this->m_EmpArray != NULL)
      {
        for (int i = 0; i < this->m_EmpNum; i++)
        {
          if (this->m_EmpArray[i] != NULL)
          {
            delete this->m_EmpArray[i];
          }
        }
        this->m_EmpNum = 0;
        delete[] this->m_EmpArray;
        this->m_EmpArray = NULL;
        this->m_FileIsEmpty = true;
      }
      cout << "清空成功！" << endl;
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

  // 标志文件是否为空
  bool m_FileIsEmpty;
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
      wm.Show_Emp();
      break;
    case 3: // 删除职工
      wm.Del_Emp();
      break;
    case 4: // 修改职工
      wm.Mod_Emp();
      break;
    case 5: // 查找职工
      wm.Find_Emp();
      break;
    case 6: // 排序职工
      wm.Sort_Emp();
      break;
    case 7: // 清空文件
      wm.Clean_File();
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
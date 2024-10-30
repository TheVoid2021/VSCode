#include <iostream>
using namespace std;

#define MAX 1000 // 最大联系人个数

// 菜单界面
void showMenu()
{
  cout << "*************************" << endl;
  cout << "*****  1.添加联系人  *****" << endl;
  cout << "*****  2.显示联系人  *****" << endl;
  cout << "*****  3.删除联系人  *****" << endl;
  cout << "*****  4.查找联系人  *****" << endl;
  cout << "*****  5.修改联系人  *****" << endl;
  cout << "*****  6.清空联系人  *****" << endl;
  cout << "*****  0.退出通讯录  *****" << endl;
  cout << "*************************" << endl;
}

// 联系人结构体
struct Person
{
  string m_Name;  // 姓名
  int m_Sex;      // 性别 1男 2女
  int m_Age;      // 年龄
  string m_Phone; // 电话
  string m_Addr;  // 住址
};

// 通讯录结构体
struct Addressbooks
{
  struct Person personArray[MAX]; // 保存联系人的数组
  int m_Size;                     // 记录通讯录中人员个数
};

// 1.添加联系人
void addPerson(struct Addressbooks *abs)
{
  // 判断通讯录是否已满
  if (abs->m_Size == MAX)
  {
    cout << "通讯录已满，无法添加" << endl;
    return;
  }
  else
  {
    // 姓名
    string name;
    cout << "请输入姓名" << endl;
    cin >> name;
    abs->personArray[abs->m_Size].m_Name = name;

    cout << "请输入性别" << endl;
    cout << "1 -- 男" << endl;
    cout << "2 -- 女" << endl;

    // 性别
    int sex = 0;
    while (true)
    {
      cin >> sex;
      if (sex == 1 || sex == 2)
      {
        abs->personArray[abs->m_Size].m_Sex = sex;
        break;
      }
      else
      {
        cout << "输入有误，请重新输入" << endl;
      }
    }

    // 年龄
    cout << "请输入年龄" << endl;
    int age = 0;
    cin >> age;
    abs->personArray[abs->m_Size].m_Age = age;

    // 电话
    cout << "请输入电话" << endl;
    string phone = "";
    cin >> phone;
    abs->personArray[abs->m_Size].m_Phone = phone;

    // 住址
    cout << "请输入住址" << endl;
    string address;
    cin >> address;
    abs->personArray[abs->m_Size].m_Addr = address;

    // 更新通讯录人数
    abs->m_Size++;

    cout << "添加成功" << endl;
    system("pause");
    system("cls");
  }
}

// 2.显示联系人
void showPerson(struct Addressbooks *abs)
{
  // 判断通讯录中是否有联系人
  if (abs->m_Size == 0)
  {
    cout << "通讯录为空" << endl;
  }
  else
  {
    for (int i = 0; i < abs->m_Size; i++)
    {
      cout << "姓名：" << abs->personArray[i].m_Name << " ";
      cout << "性别：" << (abs->personArray[i].m_Sex == 1 ? "男" : "女") << " ";
      cout << "年龄：" << abs->personArray[i].m_Age << " ";
      cout << "电话：" << abs->personArray[i].m_Phone << " ";
      cout << "住址：" << abs->personArray[i].m_Addr << endl;
    }
  }
  system("pause");
  system("cls");
}

// 判断联系人是否存在，如果存在，返回联系人下标，不存在返回-1
int isExist(struct Addressbooks *abs, string name)
{
  for (int i = 0; i < abs->m_Size; i++)
  {
    if (abs->personArray[i].m_Name == name)
    {
      return i;
    }
  }
  return -1;
}

// 3.删除联系人
void deletePerson(Addressbooks *abs)
{
  cout << "请输入要删除的联系人姓名" << endl;
  string name;
  cin >> name;

  int index = isExist(abs, name);
  if (index != -1)
  {
    for (int i = index; i < abs->m_Size; i++)
    {
      abs->personArray[i] = abs->personArray[i + 1];
    }
    abs->m_Size--;
    cout << "删除成功" << endl;
  }
  else
  {
    cout << "查无此人" << endl;
  }

  system("pause");
  system("cls");
}

// 4.查找指定联系人信息
void findPerson(Addressbooks *abs)
{
  cout << "请输入要查找的联系人姓名" << endl;
  string name;
  cin >> name;

  int index = isExist(abs, name);
  if (index != -1)
  {
    cout << "姓名：" << abs->personArray[index].m_Name << "\t";
    cout << "性别：" << (abs->personArray[index].m_Sex == 1 ? "男" : "女") << "\t";
    cout << "年龄：" << abs->personArray[index].m_Age << "\t";
    cout << "电话：" << abs->personArray[index].m_Phone << "\t";
    cout << "住址：" << abs->personArray[index].m_Addr << endl;
  }
  else
  {
    cout << "查无此人" << endl;
  }

  system("pause");
  system("cls");
}

// 5.修改指定联系人信息
void modifyPerson(Addressbooks *abs)
{
  cout << "请输入要修改的联系人姓名" << endl;
  string name;
  cin >> name;

  int index = isExist(abs, name);
  if (index != -1)
  {
    cout << "姓名：" << endl;
    string name;
    cin >> name;
    abs->personArray[index].m_Name = name;

    cout << "性别：" << endl;
    cout << "1 -- 男" << endl;
    cout << "2 -- 女" << endl;
    int sex = 0;
    while (true)
    {
      cin >> sex;
      if (sex == 1 || sex == 2)
      {
        abs->personArray[index].m_Sex = sex;
        break;
      }
      cout << "输入有误，请重新输入" << endl;
    }

    cout << "年龄：" << endl;
    int age = 0;
    cin >> age;
    abs->personArray[index].m_Age = age;

    cout << "电话：" << endl;
    string phone = "";
    cin >> phone;
    abs->personArray[index].m_Phone = phone;

    cout << "住址：" << endl;
    string address;
    cin >> address;
    abs->personArray[index].m_Addr = address;

    cout << "修改成功" << endl;
  }
  else
  {
    cout << "查无此人" << endl;
  }

  system("pause");
  system("cls");
}

// 6.清空通讯录
void cleanPerson(Addressbooks *abs)
{
  abs->m_Size = 0;
  cout << "通讯录已清空" << endl;
  system("pause");
  system("cls");
}

int main()
{
  // 创建通讯录
  struct Addressbooks abs;

  // 初始化通讯录中人数
  abs.m_Size = 0;

  int select = 0;

  while (true)
  {
    showMenu();

    cin >> select;

    switch (select)
    {
    case 1:
      cout << "添加联系人" << endl;
      addPerson(&abs);
      break;
    case 2:
      cout << "显示联系人" << endl;
      showPerson(&abs);
      break;
    case 3:
      cout << "删除联系人" << endl;
      deletePerson(&abs);
      break;
    case 4:
      cout << "查找联系人" << endl;
      findPerson(&abs);
      break;
    case 5:
      cout << "修改联系人" << endl;
      modifyPerson(&abs);
      break;
    case 6:
      cout << "清空联系人" << endl;
      cleanPerson(&abs);
      break;
    case 0:
      cout << "欢迎下次使用" << endl;
      system("pause");
      return 0;
      break;
    default:
      cout << "输入有误，请重新输入" << endl;
      break;
    }
  }
}
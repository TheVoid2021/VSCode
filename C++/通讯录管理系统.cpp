#include <iostream>
using namespace std;

#define MAX 1000 // �����ϵ�˸���

// �˵�����
void showMenu()
{
  cout << "*************************" << endl;
  cout << "*****  1.�����ϵ��  *****" << endl;
  cout << "*****  2.��ʾ��ϵ��  *****" << endl;
  cout << "*****  3.ɾ����ϵ��  *****" << endl;
  cout << "*****  4.������ϵ��  *****" << endl;
  cout << "*****  5.�޸���ϵ��  *****" << endl;
  cout << "*****  6.�����ϵ��  *****" << endl;
  cout << "*****  0.�˳�ͨѶ¼  *****" << endl;
  cout << "*************************" << endl;
}

// ��ϵ�˽ṹ��
struct Person
{
  string m_Name;  // ����
  int m_Sex;      // �Ա� 1�� 2Ů
  int m_Age;      // ����
  string m_Phone; // �绰
  string m_Addr;  // סַ
};

// ͨѶ¼�ṹ��
struct Addressbooks
{
  struct Person personArray[MAX]; // ������ϵ�˵�����
  int m_Size;                     // ��¼ͨѶ¼����Ա����
};

// 1.�����ϵ��
void addPerson(struct Addressbooks *abs)
{
  // �ж�ͨѶ¼�Ƿ�����
  if (abs->m_Size == MAX)
  {
    cout << "ͨѶ¼�������޷����" << endl;
    return;
  }
  else
  {
    // ����
    string name;
    cout << "����������" << endl;
    cin >> name;
    abs->personArray[abs->m_Size].m_Name = name;

    cout << "�������Ա�" << endl;
    cout << "1 -- ��" << endl;
    cout << "2 -- Ů" << endl;

    // �Ա�
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
        cout << "������������������" << endl;
      }
    }

    // ����
    cout << "����������" << endl;
    int age = 0;
    cin >> age;
    abs->personArray[abs->m_Size].m_Age = age;

    // �绰
    cout << "������绰" << endl;
    string phone = "";
    cin >> phone;
    abs->personArray[abs->m_Size].m_Phone = phone;

    // סַ
    cout << "������סַ" << endl;
    string address;
    cin >> address;
    abs->personArray[abs->m_Size].m_Addr = address;

    // ����ͨѶ¼����
    abs->m_Size++;

    cout << "��ӳɹ�" << endl;
    system("pause");
    system("cls");
  }
}

// 2.��ʾ��ϵ��
void showPerson(struct Addressbooks *abs)
{
  // �ж�ͨѶ¼���Ƿ�����ϵ��
  if (abs->m_Size == 0)
  {
    cout << "ͨѶ¼Ϊ��" << endl;
  }
  else
  {
    for (int i = 0; i < abs->m_Size; i++)
    {
      cout << "������" << abs->personArray[i].m_Name << " ";
      cout << "�Ա�" << (abs->personArray[i].m_Sex == 1 ? "��" : "Ů") << " ";
      cout << "���䣺" << abs->personArray[i].m_Age << " ";
      cout << "�绰��" << abs->personArray[i].m_Phone << " ";
      cout << "סַ��" << abs->personArray[i].m_Addr << endl;
    }
  }
  system("pause");
  system("cls");
}

// �ж���ϵ���Ƿ���ڣ�������ڣ�������ϵ���±꣬�����ڷ���-1
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

// 3.ɾ����ϵ��
void deletePerson(Addressbooks *abs)
{
  cout << "������Ҫɾ������ϵ������" << endl;
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
    cout << "ɾ���ɹ�" << endl;
  }
  else
  {
    cout << "���޴���" << endl;
  }

  system("pause");
  system("cls");
}

// 4.����ָ����ϵ����Ϣ
void findPerson(Addressbooks *abs)
{
  cout << "������Ҫ���ҵ���ϵ������" << endl;
  string name;
  cin >> name;

  int index = isExist(abs, name);
  if (index != -1)
  {
    cout << "������" << abs->personArray[index].m_Name << "\t";
    cout << "�Ա�" << (abs->personArray[index].m_Sex == 1 ? "��" : "Ů") << "\t";
    cout << "���䣺" << abs->personArray[index].m_Age << "\t";
    cout << "�绰��" << abs->personArray[index].m_Phone << "\t";
    cout << "סַ��" << abs->personArray[index].m_Addr << endl;
  }
  else
  {
    cout << "���޴���" << endl;
  }

  system("pause");
  system("cls");
}

// 5.�޸�ָ����ϵ����Ϣ
void modifyPerson(Addressbooks *abs)
{
  cout << "������Ҫ�޸ĵ���ϵ������" << endl;
  string name;
  cin >> name;

  int index = isExist(abs, name);
  if (index != -1)
  {
    cout << "������" << endl;
    string name;
    cin >> name;
    abs->personArray[index].m_Name = name;

    cout << "�Ա�" << endl;
    cout << "1 -- ��" << endl;
    cout << "2 -- Ů" << endl;
    int sex = 0;
    while (true)
    {
      cin >> sex;
      if (sex == 1 || sex == 2)
      {
        abs->personArray[index].m_Sex = sex;
        break;
      }
      cout << "������������������" << endl;
    }

    cout << "���䣺" << endl;
    int age = 0;
    cin >> age;
    abs->personArray[index].m_Age = age;

    cout << "�绰��" << endl;
    string phone = "";
    cin >> phone;
    abs->personArray[index].m_Phone = phone;

    cout << "סַ��" << endl;
    string address;
    cin >> address;
    abs->personArray[index].m_Addr = address;

    cout << "�޸ĳɹ�" << endl;
  }
  else
  {
    cout << "���޴���" << endl;
  }

  system("pause");
  system("cls");
}

// 6.���ͨѶ¼
void cleanPerson(Addressbooks *abs)
{
  abs->m_Size = 0;
  cout << "ͨѶ¼�����" << endl;
  system("pause");
  system("cls");
}

int main()
{
  // ����ͨѶ¼
  struct Addressbooks abs;

  // ��ʼ��ͨѶ¼������
  abs.m_Size = 0;

  int select = 0;

  while (true)
  {
    showMenu();

    cin >> select;

    switch (select)
    {
    case 1:
      cout << "�����ϵ��" << endl;
      addPerson(&abs);
      break;
    case 2:
      cout << "��ʾ��ϵ��" << endl;
      showPerson(&abs);
      break;
    case 3:
      cout << "ɾ����ϵ��" << endl;
      deletePerson(&abs);
      break;
    case 4:
      cout << "������ϵ��" << endl;
      findPerson(&abs);
      break;
    case 5:
      cout << "�޸���ϵ��" << endl;
      modifyPerson(&abs);
      break;
    case 6:
      cout << "�����ϵ��" << endl;
      cleanPerson(&abs);
      break;
    case 0:
      cout << "��ӭ�´�ʹ��" << endl;
      system("pause");
      return 0;
      break;
    default:
      cout << "������������������" << endl;
      break;
    }
  }
}
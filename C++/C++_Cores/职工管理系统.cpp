#include <iostream>
using namespace std;

/*
! ְ������ϵͳ������������˾������Ա������Ϣ
* ���̳���Ҫ����C++��ʵ��һ�����ڶ�̬��ְ������ϵͳ
? ��˾��ְ����Ϊ���ࣺ��ͨԱ���������ϰ壬��ʾ��Ϣʱ����Ҫ��ʾְ����š�ְ��������ְ����λ���Լ�ְ��
    * ��ͨԱ��ְ����ɾ�����������
    * ����ְ������ϰ彻�������񣬲��·������Ա��
    * �ϰ�ְ�𣺹���˾��������
 */

/*
! ְ���������
  * ְ���ķ���Ϊ����ͨԱ���������ϰ�
  * ������ְ������һ���ࣨworker����,���ö�̬����ְͬ������
  * ְ��������Ϊ��ְ����š�ְ��������ְ�����ڲ��ű��
  * ְ������ΪΪ����λְ����Ϣ��������ȡ��λ����*/
class Worker
{
public:
  virtual ~Worker() = default; // �������������������ͷ��������ʱ���޷�����������������������
  // ��ʾ������Ϣ
  virtual void showInfo() = 0;
  // ��ȡ��λ����
  virtual string getDeptName() = 0;

  int m_Id;      // ְ�����
  string m_Name; // ְ������
  int m_DeptId;  // ְ�����ڲ������Ʊ��
};

/*
! ��ͨԱ����
* ��ͨԱ���� �̳� ְ�������࣬����д�����д��麯��
 */
class Employee : public Worker
{
public:
  // ���캯��
  Employee(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // ��ʾ������Ϣ
  virtual void showInfo()
  {
    cout << "ְ����ţ� " << this->m_Id
         << " \tְ�������� " << this->m_Name
         << " \t��λ��" << this->getDeptName()
         << " \t��λְ����ɾ�����������" << endl;
  }

  // ��ȡְ����λ����
  virtual string getDeptName()
  {
    return string("Ա��");
  }
};

/*
! ������
* ������ �̳� ְ�������࣬����д�����д��麯��������ͨԱ������ */
class Manager : public Worker
{
public:
  Manager(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // ��ʾ������Ϣ
  virtual void showInfo()
  {
    cout << "ְ����ţ� " << this->m_Id
         << " \tְ�������� " << this->m_Name
         << " \t��λ��" << this->getDeptName()
         << " \t��λְ������ϰ彻��������,���·������Ա��" << endl;
  }

  // ��ȡְ����λ����
  virtual string getDeptName()
  {
    return string("����");
  }
};

/*
! �ϰ���
* �ϰ��� �̳� ְ�������࣬����д�����д��麯��������ͨԱ������ */
class Boss : public Worker
{
public:
  Boss(int id, string name, int dId)
  {
    this->m_Id = id;
    this->m_Name = name;
    this->m_DeptId = dId;
  }

  // ��ʾ������Ϣ
  virtual void showInfo()
  {
    cout << "ְ����ţ� " << this->m_Id
         << " \tְ�������� " << this->m_Name
         << " \t��λ��" << this->getDeptName()
         << " \t��λְ�𣺹���˾��������" << endl;
  }

  // ��ȡְ����λ����
  virtual string getDeptName()
  {
    return string("�ܲ�");
  }
};

// ! ���˹�������
class WorkerManager
{
public:
  // ���캯��
  WorkerManager()
  {
    // ��ʼ������
    this->m_EmpNum = 0;

    // ��ʼ������ָ��
    this->m_EmpArray = NULL;
  }
  void Show_Menu()
  {
    cout << "********************************************" << endl;
    cout << "*********  ��ӭʹ��ְ������ϵͳ�� **********" << endl;
    cout << "*************  0.�˳��������  *************" << endl;
    cout << "*************  1.����ְ����Ϣ  *************" << endl;
    cout << "*************  2.��ʾְ����Ϣ  *************" << endl;
    cout << "*************  3.ɾ����ְְ��  *************" << endl;
    cout << "*************  4.�޸�ְ����Ϣ  *************" << endl;
    cout << "*************  5.����ְ����Ϣ  *************" << endl;
    cout << "*************  6.���ձ������  *************" << endl;
    cout << "*************  7.��������ĵ�  *************" << endl;
    cout << "********************************************" << endl;
    cout << endl;
  }

  // �˳�ϵͳ
  void exitSystem()
  {
    cout << "��ӭ�´�ʹ��" << endl;
    system("pause");
    exit(0);
  }

  // ���ְ��
  void Add_Emp()
  {
    cout << "����������ְ�������� " << endl;

    int addNum = 0;
    cin >> addNum;

    if (addNum > 0)
    {
      // �����¿ռ��С
      int newSize = this->m_EmpNum + addNum;

      // �����¿ռ�
      Worker **newSpace = new Worker *[newSize];

      // ��ԭ�ռ������ݴ�ŵ��¿ռ���
      if (this->m_EmpArray != NULL)
      {
        for (int i = 0; i < this->m_EmpNum; i++)
        {
          newSpace[i] = this->m_EmpArray[i];
        }
      }

      // ����������
      for (int i = 0; i < addNum; i++)
      {
        int id;
        string name;
        int dSelect;

        cout << "������� " << i + 1 << " ����ְ����ţ�" << endl;
        cin >> id;

        cout << "������� " << i + 1 << " ����ְ��������" << endl;
        cin >> name;

        cout << "��ѡ���ְ���ĸ�λ��" << endl;
        cout << "1����ְͨ��" << endl;
        cout << "2������" << endl;
        cout << "3���ϰ�" << endl;
        cin >> dSelect;

        Worker *worker = NULL;
        switch (dSelect)
        {
        case 1: // ��ͨԱ��
          worker = new Employee(id, name, 1);
          break;
        case 2: // ����
          worker = new Manager(id, name, 2);
          break;
        case 3: // �ϰ�
          worker = new Boss(id, name, 3);
          break;
        default:
          break;
        }

        newSpace[this->m_EmpNum + i] = worker;
      }

      // �ͷ�ԭ�пռ�
      delete[] this->m_EmpArray;

      // �����¿ռ��ָ��
      this->m_EmpArray = newSpace;

      // �����µĸ���
      this->m_EmpNum = newSize;

      // ��ʾ��Ϣ
      cout << "�ɹ����" << addNum << "����ְ����" << endl;
    }
    else
    {
      cout << "��������" << endl;
    }

    system("pause");
    system("cls");
  }

  // ��������
  ~WorkerManager()
  {
    if (this->m_EmpArray != NULL)
    {
      delete[] this->m_EmpArray;
    }
  }

public:
  // ��¼�ļ��е���������
  int m_EmpNum;

  // Ա�������ָ��
  Worker **m_EmpArray;
};

void test()
{
  Worker *worker = NULL;
  worker = new Employee(1, "����", 1);
  worker->showInfo();
  delete worker;

  worker = new Manager(2, "����", 2);
  worker->showInfo();
  delete worker;

  worker = new Boss(3, "����", 3);
  worker->showInfo();
  delete worker;
}

int main()
{

  WorkerManager wm;

  int choice = 0;
  while (true)
  {
    // չʾ�˵�
    wm.Show_Menu();
    cout << "����������ѡ��:" << endl;
    cin >> choice;

    switch (choice)
    {
    case 0: // �˳�ϵͳ
      wm.exitSystem();
      break;
    case 1: // ���ְ��
      wm.Add_Emp();
      break;
    case 2: // ��ʾְ��
      break;
    case 3: // ɾ��ְ��
      break;
    case 4: // �޸�ְ��
      break;
    case 5: // ����ְ��
      break;
    case 6: // ����ְ��
      break;
    case 7: // ����ļ�
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
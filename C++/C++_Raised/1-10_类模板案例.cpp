#include <iostream>
using namespace std;

/*
! ��������:  ʵ��һ��ͨ�õ������࣬Ҫ�����£�
  * ���Զ��������������Լ��Զ����������͵����ݽ��д洢
  * �������е����ݴ洢������
  * ���캯���п��Դ������������
  * �ṩ��Ӧ�Ŀ������캯���Լ�operator=��ֹǳ��������
  * �ṩβ�巨��βɾ���������е����ݽ������Ӻ�ɾ��
  * ����ͨ���±�ķ�ʽ���������е�Ԫ��
  * ���Ի�ȡ�����е�ǰԪ�ظ��������������
 */

template <class T>
class MyArray
{
public:
  // ���캯��
  MyArray(int capacity)
  {
    this->m_Capacity = capacity;
    this->m_Size = 0;
    pAddress = new T[this->m_Capacity];
  }

  // ��������
  MyArray(const MyArray &arr)
  {
    this->m_Capacity = arr.m_Capacity;
    this->m_Size = arr.m_Size;
    this->pAddress = new T[this->m_Capacity];
    for (int i = 0; i < this->m_Size; i++)
    {
      // ���TΪ���󣬶��һ�����ָ�룬������Ҫ���� = ����������Ϊ����ȺŲ��� ���� ���Ǹ�ֵ��
      //  ��ͨ���Ϳ���ֱ��= ����ָ��������Ҫ���
      this->pAddress[i] = arr.pAddress[i];
    }
  }

  // ����= ������  ��ֹǳ��������
  MyArray &operator=(const MyArray &myarray)
  {

    if (this->pAddress != NULL)
    {
      delete[] this->pAddress;
      this->m_Capacity = 0;
      this->m_Size = 0;
    }

    this->m_Capacity = myarray.m_Capacity;
    this->m_Size = myarray.m_Size;
    this->pAddress = new T[this->m_Capacity];
    for (int i = 0; i < this->m_Size; i++)
    {
      this->pAddress[i] = myarray[i];
    }
    return *this;
  }

  // ����[] ������  arr[0]
  T &operator[](int index)
  {
    return this->pAddress[index]; // ������Խ�磬�û��Լ�ȥ����
  }

  // β�巨
  void Push_back(const T &val)
  {
    if (this->m_Capacity == this->m_Size)
    {
      return;
    }
    this->pAddress[this->m_Size] = val;
    this->m_Size++;
  }

  // βɾ��
  void Pop_back()
  {
    if (this->m_Size == 0)
    {
      return;
    }
    this->m_Size--;
  }

  // ��ȡ��������
  int getCapacity()
  {
    return this->m_Capacity;
  }

  // ��ȡ�����С
  int getSize()
  {
    return this->m_Size;
  }

  // ����
  ~MyArray()
  {
    if (this->pAddress != NULL)
    {
      delete[] this->pAddress;
      this->pAddress = NULL;
      this->m_Capacity = 0;
      this->m_Size = 0;
    }
  }

private:
  T *pAddress;    // ָ��һ���ѿռ䣬����ռ�洢����������
  int m_Capacity; // ����
  int m_Size;     // ��С
};

void printIntArray(MyArray<int> &arr)
{
  for (int i = 0; i < arr.getSize(); i++)
  {
    cout << arr[i] << " ";
  }
  cout << endl;
}

// ����������������
void test01()
{
  MyArray<int> array1(10);
  for (int i = 0; i < 10; i++)
  {
    array1.Push_back(i);
  }
  cout << "array1��ӡ�����" << endl;
  printIntArray(array1);
  cout << "array1�Ĵ�С��" << array1.getSize() << endl;
  cout << "array1��������" << array1.getCapacity() << endl;

  cout << "--------------------------" << endl;

  MyArray<int> array2(array1);
  array2.Pop_back();
  cout << "array2��ӡ�����" << endl;
  printIntArray(array2);
  cout << "array2�Ĵ�С��" << array2.getSize() << endl;
  cout << "array2��������" << array2.getCapacity() << endl;
}

// �����Զ�����������
class Person
{
public:
  Person() {}
  Person(string name, int age)
  {
    this->m_Name = name;
    this->m_Age = age;
  }

public:
  string m_Name;
  int m_Age;
};

void printPersonArray(MyArray<Person> &personArr)
{
  for (int i = 0; i < personArr.getSize(); i++)
  {
    cout << "������" << personArr[i].m_Name << " ���䣺 " << personArr[i].m_Age << endl;
  }
}

void test02()
{
  // ��������
  MyArray<Person> pArray(10);
  Person p1("�����", 30);
  Person p2("����", 20);
  Person p3("槼�", 18);
  Person p4("���Ѿ�", 15);
  Person p5("����", 24);

  // ��������
  pArray.Push_back(p1);
  pArray.Push_back(p2);
  pArray.Push_back(p3);
  pArray.Push_back(p4);
  pArray.Push_back(p5);

  printPersonArray(pArray);

  cout << "pArray�Ĵ�С��" << pArray.getSize() << endl;
  cout << "pArray��������" << pArray.getCapacity() << endl;
}

int main()
{

  // test01();

  test02();

  system("pause");

  return 0;
}
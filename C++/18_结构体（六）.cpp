#include <iostream>
using namespace std;

// ѧ���ṹ�嶨��
struct student
{
  // ��Ա�б�
  string name; // ����
  int age;     // ����
  int score;   // ����
};

// !constʹ�ó���
// !�õ�ַ���ݲ�����ָ���С��4���ֽڣ����ڽ�Լ�ڴ棬����Ϊ�˷�ֹ�������е�����������Լ�const����
void printStudent(const student *stu) // !��const��ֹ�������е������
{
  // stu->age = 100; //����ʧ�ܣ���Ϊ����const���� ֻ�ܶ�����д
  cout << "������" << stu->name << " ���䣺" << stu->age << " ������" << stu->score << endl;
}

int main()
{

  student stu = {"����", 18, 100};

  printStudent(&stu);

  system("pause");

  return 0;
}
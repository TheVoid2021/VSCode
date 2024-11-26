#include <iostream>
using namespace std;

/*
* C++�� *�������* �� *���ͱ��* ˼�룬Ŀ�ľ��� *�����Ե�����*
* �������£����ݽṹ���㷨��δ����һ�ױ�׼,���±��ȴ��´����ظ�����
* Ϊ�˽������ݽṹ���㷨��һ�ױ�׼,������ *STL*

! STL(Standard Template Library,<��׼ģ���>) ?
! STL �ӹ����Ϸ�Ϊ: <����(container)> <�㷨(algorithm)> <������(iterator)>
! ���� �� �㷨 ֮��ͨ�� ������ �����޷����ӡ�
! STL �������еĴ��붼������ģ�������ģ�庯��

! STL�������
? STL�����Ϊ����������ֱ���: �������㷨�����������º�����������������������ռ�������
  * ?1. �������������ݽṹ����vector��list��deque��set��map��,����������ݡ�
    * ����������֮��Ҳ
    * STL�������ǽ�������㷺��һЩ���ݽṹʵ�ֳ���
    * ���õ����ݽṹ������, ����,��, ջ, ����, ����, ӳ��� ��
    * ��Щ������Ϊ ����ʽ���� �� ����ʽ���� ����:   1 3 5 4 2 ?
      * ����ʽ����: ǿ��ֵ����������ʽ�����е�ÿ��Ԫ�ؾ��й̶���λ�á�  1 3 5 4 2
      * ����ʽ����: �������ṹ����Ԫ��֮��û���ϸ�������ϵ�˳���ϵ    1 2 3 4 5
  * ?2. �㷨�����ֳ��õ��㷨����sort��find��copy��for_each��
    * �㷨������֮�ⷨҲ
    * ���޵Ĳ��裬����߼�����ѧ�ϵ����⣬��һ��ѧ�����ǽ����㷨(Algorithms)
    * �㷨��Ϊ: �ʱ��㷨 �� ���ʱ��㷨 ��
        * �ʱ��㷨����ָ��������л���������ڵ�Ԫ�ص����ݡ����翽�����滻��ɾ���ȵ�
        * ���ʱ��㷨����ָ��������в�����������ڵ�Ԫ�����ݣ�������ҡ�������������Ѱ�Ҽ�ֵ�ȵ�
  * ?3. ���������������������㷨֮��Ľ��ϼ���
    * ���������������㷨֮��ճ�ϼ�
    * �ṩһ�ַ�����ʹ֮�ܹ�����Ѱ��ĳ�����������ĸ���Ԫ�أ��������豩¶���������ڲ���ʾ��ʽ��
    * ÿ�����������Լ�ר���ĵ�����
    * ������ʹ�÷ǳ�������ָ�룬��ѧ�׶����ǿ�������������Ϊָ��
* ���������ࣺ
| ����           | ����                                                     | ֧������                                |
| -------------- | -------------------------------------------------------- | --------------------------------------- |
| ���������     | �����ݵ�ֻ������                                         | ֻ����֧��++��==����=                   |
| ���������     | �����ݵ�ֻд����                                         | ֻд��֧��++                            |
| ǰ�������     | ��д������������ǰ�ƽ�������                             | ��д��֧��++��==����=                   |
| ?˫�������     | ��д������������ǰ��������                             | ��д��֧��++��--��                      |
| ?������ʵ����� | ��д��������������Ծ�ķ�ʽ�����������ݣ�������ǿ�ĵ����� | ��д��֧��++��--��[n]��-n��<��<=��>��>= |
    * ���õ������е���������Ϊ ˫������� �� ������ʵ�����

  * 4. �º�������Ϊ���ƺ���������Ϊ�㷨��ĳ�ֲ��ԡ�
  * 5. ��������һ�����������������߷º�����������ӿڵĶ�����
  * 6. �ռ�������������ռ�����������
 */

/*
! vector���������������
? ������     `vector`
? �㷨��     `for_each`
? �������� `vector<int>::iterator`
 */

#include <vector>    // ! vector ����ͷ�ļ�
#include <algorithm> // ! for_each �㷨ͷ�ļ�

void MyPrint(int val)
{
  cout << val << endl;
}

void test01()
{

  // todo 1.����vector�������󣬲���ͨ��ģ�����ָ�������д�ŵ����ݵ�����
  vector<int> v;
  // todo 2.�������з�����
  v.push_back(10);
  v.push_back(20);
  v.push_back(30);
  v.push_back(40);

  // ÿһ�����������Լ��ĵ����������������������������е�Ԫ��
  // todo v.begin()���� ��ʼ�����������������ָ�������е�һ������
  // todo v.end()���� ���������������������ָ������Ԫ�ص����һ��Ԫ�ص���һ��λ��
  // vector<int>::iterator �õ�vector<int>���������ĵ���������

  vector<int>::iterator pBegin = v.begin(); // pBegin�Ǹ�ָ�룬ָ�������е�һ��Ԫ��
  vector<int>::iterator pEnd = v.end();     // pEndָ�����������һ��Ԫ�ص���һ��λ��

  // todo ��һ�ֱ�����ʽ��
  while (pBegin != pEnd)
  {
    cout << *pBegin << endl; //*Ϊ�����ã�ȡ��������ָ���ֵ
    pBegin++;
  }

  // todo �ڶ��ֱ�����ʽ����forѭ����
  for (vector<int>::iterator it = v.begin(); it != v.end(); it++)
  {
    cout << *it << endl;
  }
  cout << endl;

  // todo �����ֱ�����ʽ��
  // ʹ��STL�ṩ��׼�����㷨  ͷ�ļ� algorithm
  for_each(v.begin(), v.end(), MyPrint); // for_each�㷨����һ��������ʼ���������ڶ���������������������������������
}

/*
! vector�д���Զ����������ͣ�����ӡ���
 */

// �Զ�����������
class Person
{
public:
  Person(string name, int age)
  {
    mName = name;
    mAge = age;
  }

public:
  string mName;
  int mAge;
};

// ��Ŷ���
void test02()
{

  vector<Person> v;

  // ��������
  Person p1("aaa", 10);
  Person p2("bbb", 20);
  Person p3("ccc", 30);
  Person p4("ddd", 40);
  Person p5("eee", 50);

  v.push_back(p1);
  v.push_back(p2);
  v.push_back(p3);
  v.push_back(p4);
  v.push_back(p5);

  for (vector<Person>::iterator it = v.begin(); it != v.end(); it++)
  {
    cout << "Name:" << (*it).mName << " Age:" << (*it).mAge << endl;
  }
}

// �Ŷ���ָ��
void test03()
{

  vector<Person *> v;

  // ��������
  Person p1("aaa", 10);
  Person p2("bbb", 20);
  Person p3("ccc", 30);
  Person p4("ddd", 40);
  Person p5("eee", 50);

  v.push_back(&p1); //&ȡ��ַ������p1�ĵ�ַ��ֵ��v
  v.push_back(&p2);
  v.push_back(&p3);
  v.push_back(&p4);
  v.push_back(&p5);

  for (vector<Person *>::iterator it = v.begin(); it != v.end(); it++)
  {
    Person *p = (*it);                                             // �����*it��ָ�룬*p�����ã�ȡ��ָ��ָ���ֵ��p�Ǹ�ָ��
    cout << "Name:" << p->mName << " Age:" << (*it)->mAge << endl; // p->mName �ȼ��� (*p).mName
    // ! ��չ��->��.������->������ָ��ģ�.�����ڶ����
  }
}

/*
! Vector����Ƕ������
! ������Ƕ�����������ǽ��������ݽ��б������
 */

// ����Ƕ������
void test04()
{

  vector<vector<int>> v;

  vector<int> v1;
  vector<int> v2;
  vector<int> v3;
  vector<int> v4;

  for (int i = 0; i < 4; i++)
  {
    v1.push_back(i + 1); // v1: 1 2 3 4
    v2.push_back(i + 2); // v2: 2 3 4 5
    v3.push_back(i + 3); // v3: 3 4 5 6
    v4.push_back(i + 4); // v4: 4 5 6 7
  }

  // ������Ԫ�ز��뵽vector v��
  v.push_back(v1);
  v.push_back(v2);
  v.push_back(v3);
  v.push_back(v4);

  for (vector<vector<int>>::iterator it = v.begin(); it != v.end(); it++) // �����v��һ����ά����
  {
    // ��*it��==== ���� vector<int>
    for (vector<int>::iterator vit = (*it).begin(); vit != (*it).end(); vit++) // �����*it��ָ�룬*vit�����ã�ȡ��ָ��ָ���ֵ��vit�Ǹ�ָ��
    {
      cout << *vit << " ";
    }
    cout << endl;
  }
}

int main()
{

  test01();

  test02();

  test03();

  test04();

  system("pause");

  return 0;
}
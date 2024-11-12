#include <iostream>
using namespace std;

/*
! ��̬��������������
����������
�ֱ�������ͨд���Ͷ�̬���������ʵ��������������������ļ�������

!��̬���ŵ㣺
  * ������֯�ṹ����
  * �ɶ���ǿ
  * ����ǰ�ںͺ��ڵ���չ�Լ�ά��
? C++�����ᳫ���ö�̬��Ƴ���ܹ�����Ϊ��̬�ŵ�ܶ�
 */

// ��ͨʵ��
class Calculator
{
public:
  int getResult(string oper)
  {
    if (oper == "+")
    {
      return m_Num1 + m_Num2;
    }
    else if (oper == "-")
    {
      return m_Num1 - m_Num2;
    }
    else if (oper == "*")
    {
      return m_Num1 * m_Num2;
    }
    // ���Ҫ�ṩ�µ����㣬��Ҫ�޸�Դ��
  }

public:
  int m_Num1;
  int m_Num2;
};

void test01()
{
  // ��ͨʵ�ֲ���
  Calculator c;
  c.m_Num1 = 10;
  c.m_Num2 = 10;
  cout << c.m_Num1 << " + " << c.m_Num2 << " = " << c.getResult("+") << endl;

  cout << c.m_Num1 << " - " << c.m_Num2 << " = " << c.getResult("-") << endl;

  cout << c.m_Num1 << " * " << c.m_Num2 << " = " << c.getResult("*") << endl;
}

// todo ��̬ʵ��
// todo �����������
// todo ��̬�ŵ㣺������֯�ṹ�������ɶ���ǿ������ǰ�ںͺ��ڵ���չ�Լ�ά��
class AbstractCalculator
{
public:
  /*��C++�У����һ�������麯������ô������������ҲӦ������Ϊ�麯����
  ���򣬵�ͨ������ָ��ɾ�����������ʱ��ֻ����û��������������
  ������������������������������ܻᵼ����Դй©������δ������Ϊ��? */
  virtual ~AbstractCalculator() {} // ! ����Ϊ�麯��
  virtual int getResult()
  {
    return 0;
  }

  int m_Num1;
  int m_Num2;
};

// todo �ӷ�������
class AddCalculator : public AbstractCalculator
{
public:
  int getResult()
  {
    return m_Num1 + m_Num2;
  }
};

// todo ����������
class SubCalculator : public AbstractCalculator
{
public:
  int getResult()
  {
    return m_Num1 - m_Num2;
  }
};

// todo �˷�������
class MulCalculator : public AbstractCalculator
{
public:
  int getResult()
  {
    return m_Num1 * m_Num2;
  }
};

void test02()
{
  // �����ӷ�������
  AbstractCalculator *abc = new AddCalculator;
  abc->m_Num1 = 10;
  abc->m_Num2 = 10;
  cout << abc->m_Num1 << " + " << abc->m_Num2 << " = " << abc->getResult() << endl;
  delete abc; // �����˼ǵ�����

  // ��������������
  abc = new SubCalculator;
  abc->m_Num1 = 10;
  abc->m_Num2 = 10;
  cout << abc->m_Num1 << " - " << abc->m_Num2 << " = " << abc->getResult() << endl;
  delete abc;

  // �����˷�������
  abc = new MulCalculator;
  abc->m_Num1 = 10;
  abc->m_Num2 = 10;
  cout << abc->m_Num1 << " * " << abc->m_Num2 << " = " << abc->getResult() << endl;
  delete abc;
}

int main()
{

  // test01();

  test02();

  system("pause");

  return 0;
}
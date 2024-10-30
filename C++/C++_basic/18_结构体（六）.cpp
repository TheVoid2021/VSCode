#include <iostream>
using namespace std;

// 学生结构体定义
struct student
{
  // 成员列表
  string name; // 姓名
  int age;     // 年龄
  int score;   // 分数
};

// !const使用场景
// !用地址传递参数，指针大小是4个字节，利于节约内存，但是为了防止函数体中的误操作，可以加const修饰
void printStudent(const student *stu) // !加const防止函数体中的误操作
{
  // stu->age = 100; //操作失败，因为加了const修饰 只能读不能写
  cout << "姓名：" << stu->name << " 年龄：" << stu->age << " 分数：" << stu->score << endl;
}

int main()
{

  student stu = {"张三", 18, 100};

  printStudent(&stu);

  system("pause");

  return 0;
}
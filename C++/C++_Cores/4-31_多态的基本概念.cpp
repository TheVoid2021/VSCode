#include <iostream>
using namespace std;

/*
! 多态是C++面向对象三大特性之一
? 多态分为两类
  * 静态多态: 函数重载 和 运算符重载属于静态多态，复用函数名
  * 动态多态: 派生类和虚函数实现运行时多态
? 静态多态和动态多态区别：
  * 静态多态的函数地址早绑定  -  编译阶段确定函数地址
  * 动态多态的函数地址晚绑定  -  运行阶段确定函数地址
! 多态满足条件：
! 1、有继承关系
! 2、子类重写父类中的虚函数
! 多态使用：父类指针或引用指向子类对象
! 重写：函数返回值类型  函数名 参数列表 完全一致称为重写
 */

class Animal
{
public:
  // Speak函数就是虚函数
  // ! 函数前面加上virtual关键字，变成虚函数，那么编译器在编译的时候就不能确定函数调用了。
  // ! 为了不让编译器在编译的时候确定函数调用了，那么编译器在编译的时候会将虚函数的实现地址存在虚函数表中，在运行的时候，通过虚函数表，确定函数调用的地址。
  // ! virtual虚函数关键字，vfptr虚函数(表)指针，vftable虚函数表
  virtual void speak()
  {
    cout << "动物在说话" << endl;
  }
};

class Cat : public Animal
{
public:
  // ! 子类重写父类中的虚函数，子类中的虚函数表内部会替换成子类的虚函数地址，这样在运行的时候，就会调用子类的虚函数
  void speak()
  {
    cout << "小猫在说话" << endl;
  }
};

class Dog : public Animal
{
public:
  void speak()
  {
    cout << "小狗在说话" << endl;
  }
};
// todo 我们希望传入什么对象，那么就调用什么对象的函数
// todo 如果函数地址在编译阶段就能确定，那么静态联编
// todo 如果函数地址在运行阶段才能确定，就是动态联编

// todo 地址早绑定 在编译阶段确定函数地址
void DoSpeak(Animal &animal) // todo 这里传入的是父类引用，但是传入的是子类对象
{
  animal.speak();
}

void test01()
{
  Cat cat;
  DoSpeak(cat); // !?虚函数使得这里父类的指针或者引用传入的是子类对象，但是调用的却是子类对象的函数，这就是多态

  Dog dog;
  DoSpeak(dog);
}

int main()
{

  test01();

  system("pause");

  return 0;
}
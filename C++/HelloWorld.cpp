#include <iostream>
using namespace std;
// #define Day 7

int main()
{

  // cout << "HelloWorld" << endl;

  /*标识符命名规则
   * 标识符不能是关键字
   * 标识符只能由字母 数字 下划线组成
   * 第一个字符必须为字母或下划线
   * 字母区分大小写
   */

  /*
   * 编写代码快捷键：
   * 将选定内容注释：Ctrl+K,Ctrl+C
   * 选定内容取消注释：Ctrl+K,Ctrl+U
   */

  /*1.用#define定义常量
  cout << "一周一共有" << day << "天" << endl;
  2.用const定义常量  const修饰的变量称为常量
  const int month = 12;
  cout << "一年共有" << month << "个月" << endl;*/

  // 2 4 4 8
  /*
  cout << "short 类型所占空间为：" << sizeof(short) << endl;

  cout << "int 类型所占空间为：" << sizeof(int) << endl;

  cout << "long 类型所占空间为：" << sizeof(long) << endl;

  cout << "long long 类型所占空间为：" << sizeof(long long) << endl;*/

  // 4 8
  /*float f1 = 3.14f;
  double d1 = 3.14;

  cout << f1 << endl;
  cout << d1 << endl;

  cout << "float sizeof = " << sizeof(f1) << endl;
  cout << "double sizeof =" << sizeof(d1) << endl;*/

  // 科学计数法

  float f2 = 3e2; // 3 * 10^2
  cout << "f2=" << f2 << endl;

  float f3 = 3e-2; // 3 * 0.1 ^ 2
  cout << "f3=" << f3 << endl;

  system("pause");

  return 0;
}

// VS 中常用的一些快捷键

// 一、代码自动对齐
// CTRL + K + F
//
// 二、撤销 / 反撤销
// 1、撤销-- - 使用组合键“Ctrl + Z”进行撤销操作
// 2、反撤销-- - 使用组合键“Ctrl + Y”进行反撤销操作
//
// 三、调用智能提示
// 使用组合键“Ctrl + J”或者使用组合键“Alt + →”可以在不完全输入关键词时系统自动添加提示
//
// 四、快速隐藏或显示当前代码段
// 1、ctrl + M + M
//
// 按两次M
//
// visual studio 2013 中常用的一些快捷键
//
// 五、回到上一个光标位置 / 前进到下一个光标位置
// 1、回到上一个光标位置使用组合键“Ctrl + -”
//
// 2、前进到下一个光标位置使用“Ctrl + Shift + -”
//
// 六、注释 / 取消注释
// 1、注释用组合键“Ctrl + K + C”
//
//  全注释为Ctrl+k+c，取消注释Ctrl+k+u
//
// 2、取消注释用组合键“Ctrl + K + U”
//
// 七、调试相关
// 1、设置断点-- - F9
//
// 2、启动调试-- - F5
//
// 3、逐语句调试-- - F11
//
// 4、逐过程调试-- - F10

/*
!, 红色注释
? , 蓝色注释
// , 灰色删除线注释
todo ,橘红色注释
* , 浅绿色注释
*/
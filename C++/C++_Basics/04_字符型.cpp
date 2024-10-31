#include <iostream>
using namespace std;

int main()
{ // main函数有且仅有一个

  /*
   * C和C++中字符型变量只占用一个字节
   * 字符型变量并不是单纯把字符本身放到内存中储存，而是把其对应的ASCII编码放入到储存单元
   */

  // 1、字符型变量创建方式
  char ch = 'a';
  cout << ch << endl; // 字符型变量只能一个字符，而且只能用单引号

  // 2、字符型变量所占内存大小
  cout << "char字符型变量所占内存:" << sizeof(char) << endl; // 字符串型可以有多个字符，但要用双引号

  // 3、字符型变量对应ASCII编码
  cout << (int)ch << endl; // 查看字符a对应的ASCII码
  ch = 97;                 // 可以直接用ASCII给字符型变量赋值
  cout << ch << endl;

  system("pause");

  return 0;
}
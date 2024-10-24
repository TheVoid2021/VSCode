#include <iostream>
#include <string>
using namespace std;

int main()
{

  // C风格字符串:   char 变量名[] = "字符串值"
  char str[] = "helloworld";
  cout << str << endl;
  // C++风格字符串:   string 变量名 = "字符串值"
  // 包含一个头文件 #include <string>
  string str2 = "helloworld";
  cout << str2 << endl;

  system("pause");

  return 0;
}
/*
 * continue语句
 *作用：在循环语句中，跳过本次循环中余下尚未执行的语句，继续执行下一次循环
 */
#include <iostream>
using namespace std;

int main()
{

  for (int i = 0; i < 100; i++)
  {
    if (i % 2 == 0)
    {
      continue;
    }
    cout << i << endl;
  }

  system("pause");

  return 0;
}

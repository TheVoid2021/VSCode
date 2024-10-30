#include <iostream>
using namespace std;

int main()
{
  /*
    1.查看二维数组所占内存空间
    2.获取二维数组首地址
  */

  int arr[2][3] =
      {
          {1, 2, 3},
          {4, 5, 6},
      };
  // 地址
  cout << "二维数组所占内存空间为：" << sizeof(arr) << endl;
  cout << "二维数组首地址为：" << (int *)arr << endl;
  cout << "二维数组第一行首地址为：" << (int *)arr[0] << endl;
  cout << "二维数组第二行首地址为：" << (int *)arr[1] << endl;
  cout << "二维数组第一行第一个元素地址为：" << (long long)arr[0][0] << endl;
  cout << "二维数组第二行第一个元素地址为：" << (long long)arr[1][0] << endl;

  // 大小
  cout << "二维数组一行大小为：" << sizeof(arr[0]) << endl;
  cout << "二维数组元素大小为：" << sizeof(arr[0][0]) << endl;

  cout << "二维数组行数为：" << sizeof(arr) / sizeof(arr[0]) << endl;
  cout << "二维数组列数为：" << sizeof(arr[0]) / sizeof(arr[0][0]) << endl;

  /*
  > 总结1：二维数组名就是这个数组的首地址
  > 总结2：对二维数组名进行sizeof时，可以获取整个二维数组占用的内存空间大小
  */
  system("pause");
  return 0;
}
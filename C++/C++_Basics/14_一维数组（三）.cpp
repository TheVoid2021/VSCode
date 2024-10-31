/*
* 五只小猪称体重
案例描述：
在一个数组中记录了五只小猪的体重，如：int arr[5] = {300,350,200,400,250};
找出并打印最重的小猪体重。
*/
#include <iostream>
using namespace std;

int main()
{
  // int arr[5] = { 300,350,200,400,250 };
  // int maxheight = 0;
  // for (int i = 0; i < 5; i++) {
  //	if (arr[i] > maxheight) {
  //		maxheight = arr[i];
  //	}
  // }

  // cout << "最重的小猪体重为：" << maxheight << endl;

  int weight[5];
  int heaviest = 0;

  // 输入并储存五只小猪的体重
  for (int i = 0; i < 5; i++)
  {
    cout << "请输入第" << i + 1 << "只小猪的体重：";
    cin >> weight[i];
    if (weight[i] > heaviest)
    {
      heaviest = weight[i];
    }
  }

  // 创建变量a存储有几只最重的小猪
  int a = 0;

  // 判断哪只小猪最重，并输出
  cout << "第";
  for (int i2 = 0; i2 < 5; i2++)
  {
    if (heaviest == weight[i2])
    {
      a += 1;
      // 如果同时有两只以上小猪最重，就在它们中间加顿号
      if (a > 1)
      {
        cout << "、";
      }
      cout << i2 + 1;
    }
  }
  cout << "只小猪是最重的小猪，重量为" << heaviest << "。" << endl;

  system("pause");

  return 0;
}
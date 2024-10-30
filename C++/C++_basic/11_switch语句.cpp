#include <iostream>
using namespace std;

int main()
{

  // 请给电影评分
  // 10 ~ 9   经典
  //  8 ~ 7   非常好
  //  6 ~ 5   一般
  //  5分以下 烂片

  int score = 0;
  cout << "请给电影打分" << endl;
  cin >> score;

  switch (score)
  {
  case 10:
  case 9:
    cout << "经典" << endl;
    break;
  case 8:
    cout << "非常好" << endl;
    break;
  case 7:
  case 6:
    cout << "一般" << endl;
    break;
  default: // default的作用就是switch语句里所有的case都不成立时所要执行的语句
    cout << "烂片" << endl;
    break;
  }

  system("pause");

  return 0;
}

// switch语句规则：
//
// 1、switch语句非常有用，但在使用时必须谨慎。所写的任何switch语句都必须遵循以下规则：
//
// 2、只能针对基本数据类型中的整型类型使用switch，这些类型包括int、char等。对于其他类型，则必须使用if语句。
//
// 3、switch()的参数类型不能为实型 。
//
// 4、case标签必须是常量表达式(constantExpression)，如42或者'4'。
//
// 5、case标签必须是惟一性的表达式；也就是说，不允许两个case具有相同的值。
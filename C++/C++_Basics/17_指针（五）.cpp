#include <iostream>
using namespace std;

// !值传递
void swap1(int a, int b)
{
  int temp = a;
  a = b;
  b = temp;
  //! 这里面的形参a和b变了，但是外面的实参a和b并没有变
}
// !地址传递
void swap2(int *p1, int *p2) // 把地址传给指针
{
  int temp = *p1;
  *p1 = *p2;
  *p2 = temp;
  //! 这里的p1和p2是指针，指向实参的地址，所以改变指针指向的值，这里面的形参p1和p2变了，外面的实参a和b也变了
}

int main()
{
  // *总结：如果不想修改实参，就用值传递，如果想修改实参，就用地址传递
  int a = 10;
  int b = 20;
  swap1(a, b); // !值传递不会改变实参

  swap2(&a, &b); // !地址传递会改变实参

  cout << "a = " << a << endl;

  cout << "b = " << b << endl;

  system("pause");

  return 0;
}
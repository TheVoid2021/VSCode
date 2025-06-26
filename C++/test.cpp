#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>

using namespace std;

// 定义常量
const int NUM_PARTICLES = 50;        // 粒子数量
const int MAX_ITERATIONS = 500;      // 最大迭代次数
const double INERTIA_WEIGHT = 0.7;   // 惯性权重
const double COGNITIVE_FACTOR = 1.5; // 个体学习因子
const double SOCIAL_FACTOR = 1.5;    // 群体学习因子
const double VELOCITY_LIMIT = 1.0;   // 速度限制

// 定义空间范围
const double AREA_X_MIN = 0.0;
const double AREA_X_MAX = 100.0;
const double AREA_Y_MIN = 0.0;
const double AREA_Y_MAX = 100.0;
const double AREA_Z_MIN = 0.0;
const double AREA_Z_MAX = 50.0; // 无人机飞行高度限制

// 定义无人船的固定方向（只能在网格上移动）
const int USV_GRID_STEP = 10;

// 投放点
const vector<pair<double, double>> DROP_POINTS = {{80, 20}, {50, 70}, {20, 40}}; // 投放点坐标
const pair<double, double> START_POINT = {0, 0};                                 // 无人船初始位置

// 粒子结构
struct Particle
{
  vector<double> position;     // 粒子的位置 (x, y, z)
  vector<double> velocity;     // 粒子速度
  double fitness;              // 适应度值
  vector<double> bestPosition; // 粒子自身最优位置
  double bestFitness;          // 粒子自身最优适应度
};

// 随机数生成器
double random(double min, double max)
{
  return min + (max - min) * ((double)rand() / RAND_MAX);
}

// 计算两点间的欧几里得距离
double distance(const pair<double, double> &a, const pair<double, double> &b)
{
  return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

// 适应度函数：评估路径的总成本
// 假设无人船在固定网格上移动，无人机从无人船位置到达各投放点再返回。
double fitnessFunction(const vector<pair<double, double>> &usvPath, const vector<vector<double>> &uavPaths)
{
  double totalCost = 0.0;

  // 计算无人船路径的总成本
  for (size_t i = 1; i < usvPath.size(); ++i)
  {
    totalCost += distance(usvPath[i - 1], usvPath[i]);
  }

  // 计算无人机路径的成本
  for (size_t i = 0; i < uavPaths.size(); ++i)
  {
    double droneCost = 0.0;
    pair<double, double> start = usvPath[i]; // 起点是无人船当前所在位置
    for (size_t j = 0; j < uavPaths[i].size(); j += 3)
    {
      pair<double, double> dropPoint = {uavPaths[i][j], uavPaths[i][j + 1]};
      droneCost += distance(start, dropPoint); // 起点到投放点
      start = dropPoint;
    }
    droneCost += distance(start, usvPath[i]); // 返回无人船位置
    totalCost += droneCost;
  }

  return totalCost;
}

// 更新粒子的位置和速度
void updateParticle(Particle &particle, const vector<double> &globalBestPosition)
{
  for (size_t i = 0; i < particle.position.size(); ++i)
  {
    // 更新速度
    double r1 = random(0.0, 1.0);
    double r2 = random(0.0, 1.0);
    particle.velocity[i] = INERTIA_WEIGHT * particle.velocity[i] +
                           COGNITIVE_FACTOR * r1 * (particle.bestPosition[i] - particle.position[i]) +
                           SOCIAL_FACTOR * r2 * (globalBestPosition[i] - particle.position[i]);

    // 限制速度
    particle.velocity[i] = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, particle.velocity[i]));

    // 更新位置
    particle.position[i] += particle.velocity[i];

    // 保证粒子位置在定义范围内
    if (i % 3 == 0)
    { // x 坐标
      particle.position[i] = max(AREA_X_MIN, min(AREA_X_MAX, particle.position[i]));
    }
    else if (i % 3 == 1)
    { // y 坐标
      particle.position[i] = max(AREA_Y_MIN, min(AREA_Y_MAX, particle.position[i]));
    }
    else
    { // z 坐标
      particle.position[i] = max(AREA_Z_MIN, min(AREA_Z_MAX, particle.position[i]));
    }
  }
}

int main()
{
  srand(time(0));

  // 初始化粒子群
  vector<Particle> particles(NUM_PARTICLES);
  vector<double> globalBestPosition;
  double globalBestFitness = numeric_limits<double>::max();

  for (int i = 0; i < NUM_PARTICLES; ++i)
  {
    Particle p;
    for (const auto &drop : DROP_POINTS)
    {
      p.position.push_back(random(AREA_X_MIN, AREA_X_MAX)); // x
      p.position.push_back(random(AREA_Y_MIN, AREA_Y_MAX)); // y
      p.position.push_back(random(AREA_Z_MIN, AREA_Z_MAX)); // z
      p.velocity.push_back(random(-VELOCITY_LIMIT, VELOCITY_LIMIT));
      p.velocity.push_back(random(-VELOCITY_LIMIT, VELOCITY_LIMIT));
      p.velocity.push_back(random(-VELOCITY_LIMIT, VELOCITY_LIMIT));
    }
    p.bestPosition = p.position;
    p.bestFitness = numeric_limits<double>::max();
    particles[i] = p;
  }

  // 粒子群优化主循环
  for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
  {
    for (auto &particle : particles)
    {
      // 计算适应度
      vector<pair<double, double>> usvPath = {START_POINT};
      for (const auto &drop : DROP_POINTS)
      {
        usvPath.push_back({particle.position[0], particle.position[1]});
      }

      vector<vector<double>> uavPaths;
      for (size_t i = 0; i < DROP_POINTS.size(); ++i)
      {
        vector<double> uavPath;
        for (size_t j = 0; j < 3; ++j)
        {
          uavPath.push_back(particle.position[i * 3 + j]);
        }
        uavPaths.push_back(uavPath);
      }

      particle.fitness = fitnessFunction(usvPath, uavPaths);

      // 更新个体最优
      if (particle.fitness < particle.bestFitness)
      {
        particle.bestFitness = particle.fitness;
        particle.bestPosition = particle.position;
      }

      // 更新全局最优
      if (particle.fitness < globalBestFitness)
      {
        globalBestFitness = particle.fitness;
        globalBestPosition = particle.position;
      }
    }

    // 更新粒子
    for (auto &particle : particles)
    {
      updateParticle(particle, globalBestPosition);
    }

    // 输出当前迭代的最优值
    cout << "Iteration " << iter + 1 << " Best Fitness: " << globalBestFitness << endl;
  }

  cout << "Optimization completed!" << endl;
  cout << "Best Fitness: " << globalBestFitness << endl;

  return 0;
}

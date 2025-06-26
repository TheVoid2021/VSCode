#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <algorithm>

using namespace std;

// ���峣��
const int NUM_PARTICLES = 50;        // ��������
const int MAX_ITERATIONS = 500;      // ����������
const double INERTIA_WEIGHT = 0.7;   // ����Ȩ��
const double COGNITIVE_FACTOR = 1.5; // ����ѧϰ����
const double SOCIAL_FACTOR = 1.5;    // Ⱥ��ѧϰ����
const double VELOCITY_LIMIT = 1.0;   // �ٶ�����

// ����ռ䷶Χ
const double AREA_X_MIN = 0.0;
const double AREA_X_MAX = 100.0;
const double AREA_Y_MIN = 0.0;
const double AREA_Y_MAX = 100.0;
const double AREA_Z_MIN = 0.0;
const double AREA_Z_MAX = 50.0; // ���˻����и߶�����

// �������˴��Ĺ̶�����ֻ�����������ƶ���
const int USV_GRID_STEP = 10;

// Ͷ�ŵ�
const vector<pair<double, double>> DROP_POINTS = {{80, 20}, {50, 70}, {20, 40}}; // Ͷ�ŵ�����
const pair<double, double> START_POINT = {0, 0};                                 // ���˴���ʼλ��

// ���ӽṹ
struct Particle
{
  vector<double> position;     // ���ӵ�λ�� (x, y, z)
  vector<double> velocity;     // �����ٶ�
  double fitness;              // ��Ӧ��ֵ
  vector<double> bestPosition; // ������������λ��
  double bestFitness;          // ��������������Ӧ��
};

// �����������
double random(double min, double max)
{
  return min + (max - min) * ((double)rand() / RAND_MAX);
}

// ����������ŷ����þ���
double distance(const pair<double, double> &a, const pair<double, double> &b)
{
  return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
}

// ��Ӧ�Ⱥ���������·�����ܳɱ�
// �������˴��ڹ̶��������ƶ������˻������˴�λ�õ����Ͷ�ŵ��ٷ��ء�
double fitnessFunction(const vector<pair<double, double>> &usvPath, const vector<vector<double>> &uavPaths)
{
  double totalCost = 0.0;

  // �������˴�·�����ܳɱ�
  for (size_t i = 1; i < usvPath.size(); ++i)
  {
    totalCost += distance(usvPath[i - 1], usvPath[i]);
  }

  // �������˻�·���ĳɱ�
  for (size_t i = 0; i < uavPaths.size(); ++i)
  {
    double droneCost = 0.0;
    pair<double, double> start = usvPath[i]; // ��������˴���ǰ����λ��
    for (size_t j = 0; j < uavPaths[i].size(); j += 3)
    {
      pair<double, double> dropPoint = {uavPaths[i][j], uavPaths[i][j + 1]};
      droneCost += distance(start, dropPoint); // ��㵽Ͷ�ŵ�
      start = dropPoint;
    }
    droneCost += distance(start, usvPath[i]); // �������˴�λ��
    totalCost += droneCost;
  }

  return totalCost;
}

// �������ӵ�λ�ú��ٶ�
void updateParticle(Particle &particle, const vector<double> &globalBestPosition)
{
  for (size_t i = 0; i < particle.position.size(); ++i)
  {
    // �����ٶ�
    double r1 = random(0.0, 1.0);
    double r2 = random(0.0, 1.0);
    particle.velocity[i] = INERTIA_WEIGHT * particle.velocity[i] +
                           COGNITIVE_FACTOR * r1 * (particle.bestPosition[i] - particle.position[i]) +
                           SOCIAL_FACTOR * r2 * (globalBestPosition[i] - particle.position[i]);

    // �����ٶ�
    particle.velocity[i] = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, particle.velocity[i]));

    // ����λ��
    particle.position[i] += particle.velocity[i];

    // ��֤����λ���ڶ��巶Χ��
    if (i % 3 == 0)
    { // x ����
      particle.position[i] = max(AREA_X_MIN, min(AREA_X_MAX, particle.position[i]));
    }
    else if (i % 3 == 1)
    { // y ����
      particle.position[i] = max(AREA_Y_MIN, min(AREA_Y_MAX, particle.position[i]));
    }
    else
    { // z ����
      particle.position[i] = max(AREA_Z_MIN, min(AREA_Z_MAX, particle.position[i]));
    }
  }
}

int main()
{
  srand(time(0));

  // ��ʼ������Ⱥ
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

  // ����Ⱥ�Ż���ѭ��
  for (int iter = 0; iter < MAX_ITERATIONS; ++iter)
  {
    for (auto &particle : particles)
    {
      // ������Ӧ��
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

      // ���¸�������
      if (particle.fitness < particle.bestFitness)
      {
        particle.bestFitness = particle.fitness;
        particle.bestPosition = particle.position;
      }

      // ����ȫ������
      if (particle.fitness < globalBestFitness)
      {
        globalBestFitness = particle.fitness;
        globalBestPosition = particle.position;
      }
    }

    // ��������
    for (auto &particle : particles)
    {
      updateParticle(particle, globalBestPosition);
    }

    // �����ǰ����������ֵ
    cout << "Iteration " << iter + 1 << " Best Fitness: " << globalBestFitness << endl;
  }

  cout << "Optimization completed!" << endl;
  cout << "Best Fitness: " << globalBestFitness << endl;

  return 0;
}

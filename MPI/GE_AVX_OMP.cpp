#include<iostream>
#include<fstream>
#include<cmath>
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>
#include<immintrin.h>
#include<omp.h>

#define PROGRESS_NUM 8
#define THREAD_NUM 8

const int n = 1000;
float m[n][n];

void show(int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			std::cout << m[i][j] << " ";
		std::cout << std::endl;
	}
}
/*
void* PT_Static_Div_Elem_AVX(void* param) {  // 三重循环全部纳入
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // 使用浮点数暂存数据以减少程序中地址的访问次数
	__m256 va, vt, vaik, vakj, vaij, vx;
	int j = 0;
	for (int k = 0; k < n; ++k) {
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		// 比信号量更简洁的同步方式是使用 barrier
		if (t_id == 0)
		{
			vt = _mm256_set1_ps(mat[k][k]);  // 对除法算法进行SIMD并行化
			t1 = mat[k][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&mat[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&mat[k][j], va);
			}
			for (j; j < n; j++)
			{
				mat[k][j] = mat[k][j] / t1;  // 善后
			}
			mat[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Divsion[i]);
			}
		}

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM) {
			//消去
			vaik = _mm256_set1_ps(mat[i][k]);
			t2 = mat[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&mat[k][j]);
				vaij = _mm256_loadu_ps(&mat[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&mat[i][j], vaij);
			}
			for (j; j < n; j++)
			{
				mat[i][j] -= t2 * mat[k][j];
			}
			mat[i][k] = 0;
		}
		if (t_id == 0) {
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			}
			for (int i = 0; i < THREAD_NUM - 1; i++) {
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
			}
		}
		else {
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}*/

float** generate(int n) {
	int k = 0;
	std::ifstream inp("input.txt");
	inp >> k;
	float** m = new float* [n];
	for (int i = 0; i < n; i++)
	{
		m[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			inp >> m[i][j];
		}
	}
	inp.close();
	return m;
}

int getEnd(int rank)
{
	int t_end;
	if (rank == PROGRESS_NUM - 1) t_end = n - 1;
	else t_end = (rank + 1) * (n / PROGRESS_NUM) - 1;
	return t_end;
}

int main(int argc, char* argv[])
{
	std::ifstream inp("input.txt");
	int pn;
	inp >> pn;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			inp >> m[i][j];

	//m1 = generate(n);  // 使用类似以往的动态初始化可能会出现问题，一般使用静态

	//show(n);  // 展示初始化结果

	MPI_Init(0, 0);
	int rank;
	//MPI_Status status;
	double head, tail;
	head = MPI_Wtime();

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// 块划分
	int t_begin = rank * (n / PROGRESS_NUM);
	int t_end = getEnd(rank);

	if (rank == 0) {
		// 分工
		int tasknum, t1, t2;
		for (int j = 1; j < PROGRESS_NUM; j++)
		{
			t1 = j * (n / PROGRESS_NUM);
			t2 = getEnd(j);
			tasknum = n * (t2 - t1 + 1);
			MPI_Send(&m[t1][0], tasknum, MPI_FLOAT, j, n + 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&m[t_begin][0], n * (t_end - t_begin + 1), MPI_FLOAT, 0, n + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	float d_temp, e_temp;  // 使用d_temp（除法temp）和e_temp（消去temp）来减少跳跃的内存访问次数
	int i, k, j = 0;
	__m256 va, vt, vaik, vakj, vaij, vx;
#pragma omp parallel if(1), num_threads(THREAD_NUM), private(i, j, k, va, vt, vaik, vakj, vaij, vx, d_temp, e_temp)
	for (int k = 0; k < n; k++)
	{
		if (rank == 0)
		{	//0号进程负责除法
			vt = _mm256_set1_ps(m[k][k]);  // 对除法算法进行SIMD并行化
			d_temp = m[k][k];

			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&m[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&m[k][j], va);
			}
			for (j; j < n; j++)
			{
				m[k][j] = m[k][j] / d_temp;  // 善后
			}
			m[k][k] = 1.0;
			for (int j = 1; j < PROGRESS_NUM; j++)
			{
				MPI_Send(&m[k][0], n, MPI_FLOAT, j, k + 1, MPI_COMM_WORLD);
			}
		}
		else
		{
			MPI_Recv(&m[k][0], n, MPI_FLOAT, 0, k + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (t_end > k)
		{
			for (int i = std::max(k + 1, t_begin); i <= t_end; i++)
			{  // 防溢出，块划分思想
				vaik = _mm256_set1_ps(m[i][k]);
				e_temp = m[i][k];
				for (j = k + 1; j + 8 < n; j += 8)
				{
					vakj = _mm256_loadu_ps(&m[k][j]);
					vaij = _mm256_loadu_ps(&m[i][j]);
					vx = _mm256_mul_ps(vakj, vaik);
					vaij = _mm256_sub_ps(vaij, vx);
					_mm256_storeu_ps(&m[i][j], vaij);
				}
				for (j; j < n; j++)
				{
					m[i][j] -= e_temp * m[k][j];
				}
				m[i][k] = 0;
				if (i == k + 1 && rank)
				{
					MPI_Send(&m[i][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
		}
		if (!rank && t_end < k + 1 && n > k + 1)
		{
			MPI_Recv(&m[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tail = MPI_Wtime();

	MPI_Finalize();
	if (!rank)
	{
		std::cout << "MPI_GE_Cycle：" << (tail - head) * 1000 << " ms" << std::endl;
		//show(n);
	}
	return 0;

}
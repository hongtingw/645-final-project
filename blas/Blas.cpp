#include "Blas.h"
#include <glog/logging.h>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

cv::Mat Blas::multiplyOpenCV(const cv::Mat &a, const cv::Mat &b) {
	return a * b;
}

cv::Mat Blas::multiplyOptimized(const cv::Mat &a, const cv::Mat &b) {
	// Optimize 2D matrix multiplication here.
	cv::Mat nb = b.t();
	float* mata = (float*)&a.at<float>(0, 0);
	float* matb = (float*)&nb.at<float>(0, 0);
	cv::Mat c = cv::Mat::zeros(cv::Size(b.cols, a.rows), a.type());
	__m128 X, Y, acc;
	float result, temp[4];
#pragma omp parallel for private(X, Y, acc, result, temp)
	for (unsigned int i = 0; i < a.rows; i++)
	{
		unsigned int begin1 = i*a.cols;
		for (unsigned int j = 0; j < b.cols; j++)
		{
			result = 0;
			acc = _mm_setzero_ps();
			unsigned int begin2 = j*b.rows;
			unsigned int k;
			for (k = 0; k < a.cols - 4; k += 4)
			{
				X = _mm_loadu_ps(mata + begin1 + k);
				Y = _mm_loadu_ps(matb + begin2 + k);
				acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
			}
			_mm_storeu_ps(temp, acc);
			result += temp[0] + temp[1] + temp[2] + temp[3];
			for (; k < a.cols; k++)
				result += mata[begin1 + k] * matb[begin2 + k];
			c.at<float>(i, j) = result;
		}
	}
	return c;
}

cv::Mat Blas::multiplyNaive(const cv::Mat &a, const cv::Mat &b) {
	CHECK_EQ(a.cols, b.rows) << "Cannot multiply matrices with size " << a.size() << " and " << b.size() << "!";
	CHECK_EQ(a.type(), CV_32F) << "Matrix A is of unsupported type!";
	CHECK_EQ(b.type(), CV_32F) << "Matrix B is of unsupported type!";
	cv::Mat c = cv::Mat::zeros(cv::Size(b.cols, a.rows), a.type());
	for (int i = 0; i < a.rows; ++i) {
		for (int j = 0; j < b.cols; ++j) {
			for (int k = 0; k < a.cols; ++k) {
				c.at<float>(i, j) += a.at<float>(i, k) * b.at<float>(k, j);
			}
		}
	}
	return c;
}

cv::Mat Blas::multiply(const cv::Mat &a, const cv::Mat &b) {
	switch (mat_op_impl_) {
	case NAIVE:return multiplyNaive(a, b);
	case OPENCV:return multiplyOpenCV(a, b);
	case OPT:return multiplyOptimized(a, b);
	}
}

cv::Mat Blas::addOpenCV(const cv::Mat &a, const cv::Mat &b) {
	return a + b;
}

cv::Mat Blas::addOptimized(const cv::Mat &a, const cv::Mat &b) {
	float* mata = (float*)&a.at<float>(0, 0);
	float* matb = (float*)&b.at<float>(0, 0);
	cv::Mat c = cv::Mat::zeros(cv::Size(a.cols, a.rows), a.type());
	__m128 X, Y, acc;
	float result, temp[4];
#pragma omp parallel for private(X, Y, acc, result, temp)
	for (unsigned int i = 0; i < a.rows; i++)
	{
		unsigned int begin = i*a.cols;
		unsigned int j;
		for (j = 0; j + 4 < a.cols; j += 4)
		{
			X = _mm_loadu_ps(mata + begin + j);
			Y = _mm_loadu_ps(matb + begin + j);
			acc = _mm_add_ps(X, Y);
			_mm_storeu_ps(temp, acc);
			c.at<float>(i, j) = temp[0];
			c.at<float>(i, j + 1) = temp[1];
			c.at<float>(i, j + 2) = temp[2];
			c.at<float>(i, j + 3) = temp[3];
		}
		for (; j < a.cols; j++)
			c.at<float>(i, j) = a.at<float>(i, j) + b.at<float>(i, j);
	}
	return c;
}

cv::Mat Blas::addNaive(const cv::Mat &a, const cv::Mat &b) {
	CHECK_EQ(a.size(), b.size()) << "Matrix sizes do not match!";
	CHECK_EQ(a.type(), CV_32F) << "Matrix A is of unsupported type!";
	CHECK_EQ(b.type(), CV_32F) << "Matrix B is of unsupported type!";
	cv::Mat c(cv::Size(a.cols, a.rows), a.type());
	for (int i = 0; i < a.rows; ++i) {
		for (int j = 0; j < b.cols; ++j) {
			c.at<float>(i, j) = a.at<float>(i, j) + b.at<float>(i, j);
		}
	}
	return c;
}

cv::Mat Blas::add(const cv::Mat &a, const cv::Mat &b) {
	switch (mat_op_impl_) {
	case NAIVE:return addNaive(a, b);
	case OPENCV:return addOpenCV(a, b);
	case OPT:return addOptimized(a, b);
	}
}
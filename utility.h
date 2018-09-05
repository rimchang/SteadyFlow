#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Eigen>
#include <iostream>
#include <cv.h>
#include <fstream>

//#include <glog/logging.h>
using namespace Eigen;
using namespace cv;
using namespace std;

namespace interpolation_util {
	template<typename T, int N>
	Eigen::Matrix<double, N, 1> bilinear(const T *const data, const int w, const int h, const Eigen::Vector2d &loc) {
		using namespace Eigen;
		const double epsilon = 0.00001;
		int xl = floor(loc[0] - epsilon), xh = (int) round(loc[0] + 0.5 - epsilon);
		int yl = floor(loc[1] - epsilon), yh = (int) round(loc[1] + 0.5 - epsilon);

		if (loc[0] <= epsilon)
			xl = 0;
		if (loc[1] <= epsilon)
			yl = 0;

		const int l1 = yl * w + xl;
		const int l2 = yh * w + xh;
		if (l1 == l2) {
			Matrix<double, N, 1> res;
			for (size_t i = 0; i < N; ++i)
				res[i] = data[l1 * N + i];
			return res;
		}

//	char buffer[100] = {};
//		sprintf(buffer, "bilinear(): coordinate out of range: (%.2f,%.2f), (%d,%d,%d,%d), l1:%d, l2:%d!",
//		        loc[0], loc[1], xl, yl, xh, yh, l1, l2);
		//CHECK(!(l1 < 0 || l2 < 0 || l1 >= w * h || l2 >= w * h)) << loc[0] << ' ' << loc[1] << ' '<< w << ' '<< h;

		double lm = loc[0] - (double) xl, rm = (double) xh - loc[0];
		double tm = loc[1] - (double) yl, bm = (double) yh - loc[1];

		// (xl, yl), (xh,yl) , (xh,yh), (xl, yh)
		Vector4i ind(xl + yl * w, xh + yl * w, xh + yh * w, xl + yh * w);

		// N채널의 픽셀값.
		std::vector<Matrix<double, N, 1> > v(4);
		for (size_t i = 0; i < 4; ++i) {
			for (size_t j = 0; j < N; ++j)
				v[i][j] = data[ind[i] * N + j];
		}

		// epsilon보다 작은 차이면 이런일이 발생한다.
		if (std::abs(lm) <= epsilon && std::abs(rm) <= epsilon)
			return (v[0] * bm + v[2] * tm) / (bm + tm);

		if (std::abs(bm) <= epsilon && std::abs(tm) <= epsilon)
			return (v[0] * rm + v[2] * lm) / (lm + rm);

		Vector4d vw(rm * bm, lm * bm, lm * tm, rm * tm);
		double sum = vw.sum();
//	sprintf(buffer, "loc:(%.2f,%.2f), integer: (%d,%d,%d,%d), margin: (%.2f,%.2f,%.2f,%.2f), sum: %.2f",
//		loc[0], loc[1], xl, yl, xh, yh, lm, rm, tm, bm, sum);
		//CHECK_GT(sum, 0);
		return (v[0] * vw[0] + v[1] * vw[1] + v[2] * vw[2] + v[3] * vw[3]) / sum;
	};
}

namespace math_util {
	inline double variance(const std::vector<double> &a, const double mean) {
		//CHECK_GT(a.size(),1);
		const double n = (double) a.size();
		std::vector<double> diff(a.size());
		std::transform(a.begin(), a.end(), diff.begin(), std::bind2nd(std::minus<double>(), mean));
		return std::sqrt(std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0) / (n-1));
	}

	inline double variance(const std::vector<double> &a) {
		//CHECK(!a.empty());
		const double mean = std::accumulate(a.begin(), a.end(), 0.0) / (double) a.size();
		return variance(a, mean);
	}

}//namespace math_util

namespace visualization_util{

	inline bool isFlowCorrect(Point2f u)
	{
		return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
	}

	Vec3b computeColor(float fx, float fy);
	void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1);

}

namespace vectorMat_util {
	inline void vecmatwrite(const string& filename, const vector<Mat>& matrices)
		{
			ofstream fs(filename.c_str(), fstream::binary);
			cout << "save vector<Mat> to" << filename << endl;
			for (size_t i = 0; i < matrices.size(); ++i)
			{
				const Mat& mat = matrices[i];

				// Header
				int type = mat.type();
				int channels = mat.channels();
				fs.write((char*)&mat.rows, sizeof(int));    // rows
				fs.write((char*)&mat.cols, sizeof(int));    // cols
				fs.write((char*)&type, sizeof(int));        // type
				fs.write((char*)&channels, sizeof(int));    // channels

				// Data
				if (mat.isContinuous())
				{
					fs.write((char*)mat.ptr<Mat>(0), (mat.dataend - mat.datastart));
				}
				else
				{
					int rowsz = CV_ELEM_SIZE(type) * mat.cols;
					for (int r = 0; r < mat.rows; ++r)
					{
						fs.write((char*)mat.ptr<Mat>(r), rowsz);
					}
				}
			}
		}

	inline vector<Mat> vecmatread(const string& filename)
		{
			vector<Mat> matrices;
			ifstream fs(filename.c_str(), fstream::binary);

			// Get length of file
			fs.seekg(0, fs.end);
			int length = fs.tellg();
			fs.seekg(0, fs.beg);

			while (fs.tellg() < length)
			{
				// Header
				int rows, cols, type, channels;
				fs.read((char*)&rows, sizeof(int));         // rows
				fs.read((char*)&cols, sizeof(int));         // cols
				fs.read((char*)&type, sizeof(int));         // type
				fs.read((char*)&channels, sizeof(int));     // channels

				// Data

				Mat mat(rows, cols, type);
				fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

				matrices.push_back(mat);
			}
			return matrices;
		}

}

#endif

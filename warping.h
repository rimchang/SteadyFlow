//
// Created by yanhang on 4/21/16.
//

#ifndef WARPPING_H
#define WARPPING_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <iostream>
//#include <glog/logging.h>

namespace Warping {

	class GridWarpping {
	public:
		GridWarpping(const int w, const int h, const int gw = 40, const int gh = 40);

		void getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind, Eigen::Vector4d &w) const;

		void getWarpedGridIndAndWeight(const std::vector<Eigen::Vector2d> &grid, const Eigen::Vector2d &pt, Eigen::Vector4i &ind, Eigen::Vector4d &w) const;

		bool validGridCheck(const std::vector<Eigen::Vector2d> &grid, const Eigen::Vector4i &ind, const Eigen::Vector2d &pt) const;

		Eigen::Vector2d getLocalCoord(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3) const;

		void warpImageCloseForm(const cv::Mat &input, cv::Mat &output, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2) const;

		void addCoefficient(std::vector<Eigen::Triplet<double> > &triplets, Eigen::VectorXd &B, const int x, const int y, int &cInd, const int condition) const;

		void computeSaliencyWeight(const cv::Mat& input);

		inline int gridInd(int x, int y)const{
			//CHECK_LE(x, gridW);
			//CHECK_LE(y, gridH);
			return y*(gridW+1)+x;
		}

		inline int sailencyInd(int x, int y)const{
			//CHECK_LE(x, gridW);
			//CHECK_LE(y, gridH);
			return y*(gridW)+x;
		}
		inline double getBlockW() const{
			return blockW;
		}
		inline double getBlockH() const{
			return blockH;
		}

		void visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat& img) const;

	  void setWdata(double val) { wdata_ = val; }
	  double wdata() const { return wdata_; }

	  void setAlpha(double val) { alpha_ = val; }
	  double alpha() const { return alpha_; }

	private:
		std::vector<Eigen::Vector2d> gridLoc;
		std::vector<double> saliency;
		int width;
		int height;
		int gridW;
		int gridH;
		double blockW;
		double blockH;

		double wdata_;
		double alpha_;
	};

}//namespace substab

#endif //SUBSPACESTAB_WARPPING_H

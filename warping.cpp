#include "warping.h"
//#include "gridenergy.h"
#include "utility.h"


#include <Eigen/Sparse>
//#include <Eigen/SparseQR>
#include <Eigen/SPQRSupport>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

namespace Warping{

	GridWarpping::GridWarpping(const int w, const int h, const int gw, const int gh)
	: width(w), height(h), gridW(gw), gridH(gh) {
		blockW = (double) width / gridW;
		blockH = (double) height / gridH;
		gridLoc.resize((size_t) (gridW + 1) * (gridH + 1));
		for (auto x = 0; x <= gridW; ++x) {
			for (auto y = 0; y <= gridH; ++y) {
				gridLoc[y * (gridW + 1) + x] = Eigen::Vector2d(blockW * x, blockH * y);


				if (x == gridW)
					gridLoc[y * (gridW + 1) + x][0] -= 1.1;
				if (y == gridH)
					gridLoc[y * (gridW + 1) + x][1] -= 1.1;

			}
		}

		setWdata(1.0);
		setAlpha(20.0);
		saliency.resize((size_t)(gridW*gridH), 1.0);
	}


	bool GridWarpping::validGridCheck(const std::vector<Eigen::Vector2d> &grid, const Eigen::Vector4i &ind, const Eigen::Vector2d &pt) const {

		const double &xd = pt[0];
		const double &yd = pt[1];


		const Vector2d vertex1 = grid[ind[0]];
		const Vector2d vertex2 = grid[ind[1]];
		const Vector2d vertex3 = grid[ind[2]];
		const Vector2d vertex4 = grid[ind[3]];

		bool valid_x = vertex1[0] <= xd && vertex4[0] <= xd && vertex2[0] >= xd && vertex3[0] >= xd;
		bool valid_y = vertex1[1] <= yd && vertex2[1] <= yd && vertex4[1] >= yd && vertex3[1] >= yd;

		return valid_x && valid_y;

	}

	void GridWarpping::getWarpedGridIndAndWeight(const std::vector<Eigen::Vector2d> &warpedgrid, const Eigen::Vector2d &pt,
															Eigen::Vector4i &ind, Eigen::Vector4d &w) const {

		int x = (int) floor(pt[0] / blockW);
		int y = (int) floor(pt[1] / blockH);

		int x_;
		int y_;

		//////////////
		// 1--2
		// |  |
		// 4--3
		/////////////
		const double &xd = pt[0];
		const double &yd = pt[1];

			//cout << "un_valid" << endl;
			for (int i=-10; i<=10; ++i){
				for (int j=-10; j<=10; ++j) {

					x_ = x + i;
					y_ = y + j;

					if (x_ <= 0 || y_ <= 0 || x_ >= gridW || y_ >= gridH) {
						continue;
					}

					Vector4i ori_ind = Vector4i(y_ * (gridW + 1) + x_, y_ * (gridW + 1) + x_ + 1, (y_ + 1) * (gridW + 1) + x_ + 1,
								   (y_ + 1) * (gridW + 1) + x_);

					if (validGridCheck(warpedgrid, ori_ind, pt)) {

						ind = ori_ind;

						const double xl = gridLoc[ind[0]][0];
						const double xh = gridLoc[ind[2]][0];
						const double yl = gridLoc[ind[0]][1];
						const double yh = gridLoc[ind[2]][1];

						w[0] = (xh - xd) * (yh - yd);
						w[1] = (xd - xl) * (yh - yd);
						w[2] = (xd - xl) * (yd - yl);
						w[3] = (xh - xd) * (yd - yl);

						double s = w[0] + w[1] + w[2] + w[3];
						//CHECK_GT(s, 0) << pt[0] << ' '<< pt[1];
						w = w / s;
						//cout<< w << "in funcuntion"<<endl;
						//cout << "find valid grid" << endl;

						return;
					}
				}
			}

	}

	Vector2d GridWarpping::getLocalCoord(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2, const Eigen::Vector2d& p3) const {

		Vector2d axis1 = p3 - p2;
		Vector2d axis2(-1*axis1[1], axis1[0]);
		Vector2d v = p1 - p2;

		return Vector2d(v.dot(axis1)/axis1.squaredNorm(), v.dot(axis2)/axis2.squaredNorm());

	}

	void GridWarpping::getGridIndAndWeight(const Eigen::Vector2d &pt, Eigen::Vector4i &ind,
										   Eigen::Vector4d &w) const {
		//CHECK_LE(pt[0], width - 1);
		//CHECK_LE(pt[1], height - 1);
		int x = (int)floor(pt[0] / blockW);
		int y = (int)floor(pt[1] / blockH);

		//////////////
		// 1--2
		// |  |
		// 4--3
		/////////////
		ind = Vector4i(y * (gridW + 1) + x, y * (gridW + 1) + x + 1, (y + 1) * (gridW + 1) + x + 1,
					   (y + 1) * (gridW + 1) + x);

		const double &xd = pt[0];
		const double &yd = pt[1];
		const double xl = gridLoc[ind[0]][0];
		const double xh = gridLoc[ind[2]][0];
		const double yl = gridLoc[ind[0]][1];
		const double yh = gridLoc[ind[2]][1];

		w[0] = (xh - xd) * (yh - yd);
		w[1] = (xd - xl) * (yh - yd);
		w[2] = (xd - xl) * (yd - yl);
		w[3] = (xh - xd) * (yd - yl);

		double s = w[0] + w[1] + w[2] + w[3];
		//CHECK_GT(s, 0) << pt[0] << ' '<< pt[1];
		w = w / s;

		Vector2d pt2 =
				gridLoc[ind[0]] * w[0] + gridLoc[ind[1]] * w[1] + gridLoc[ind[2]] * w[2] + gridLoc[ind[3]] * w[3];
		double error = (pt2 - pt).norm();
		//CHECK_LT(error, 0.0001) << pt[0] << ' ' << pt[1] << ' ' << pt2[0] << ' ' << pt2[1];
	}


	void GridWarpping::visualizeGrid(const std::vector<Eigen::Vector2d>& grid, cv::Mat &img) const {
		//CHECK_EQ(grid.size(), gridLoc.size());
		//CHECK_EQ(img.cols, width);
		//CHECK_EQ(img.rows, height);
		//img = Mat(height, width, CV_8UC3, Scalar(0,0,0));
		for(auto gy=0; gy<gridH; ++gy) {
			for (auto gx = 0; gx < gridW; ++gx){
				const int gid1 = gy * (gridW+1) + gx;
				const int gid2 = (gy+1) * (gridW+1) + gx;
				const int gid3 = (gy+1)*(gridW+1)+gx+1;
				const int gid4= gy * (gridW+1) + gx+1;
				if(grid[gid1][0] > 0 && grid[gid2][0] > 0)
					cv::line(img, cv::Point(grid[gid1][0], grid[gid1][1]), cv::Point(grid[gid2][0], grid[gid2][1]), Scalar(255,255,255));
				if(grid[gid2][0] > 0 && grid[gid3][0] > 0)
					cv::line(img, cv::Point(grid[gid2][0], grid[gid2][1]), cv::Point(grid[gid3][0], grid[gid3][1]), Scalar(255,255,255));
				if(grid[gid3][0] > 0 && grid[gid4][0] > 0)
					cv::line(img, cv::Point(grid[gid3][0], grid[gid3][1]), cv::Point(grid[gid4][0], grid[gid4][1]), Scalar(255,255,255));
				if(grid[gid4][0] > 0 && grid[gid1][0] > 0)
					cv::line(img, cv::Point(grid[gid4][0], grid[gid4][1]), cv::Point(grid[gid1][0], grid[gid1][1]), Scalar(255,255,255));
			}
		}
	}

	void GridWarpping::computeSaliencyWeight(const cv::Mat &input) {
		// Point2f(x,y) 기준으로 오른쪽 밑의 grid에 대한 w_s
		//saliency.resize((size_t)(gridW*gridH));
		for(auto y=0; y<gridH; ++y){
			for(auto x=0; x<gridW; ++x){
				const int sid = sailencyInd(x,y);
				vector<vector<double> > pixs(3);
				for(auto x1=(int)gridLoc[gridInd(x,y)][0]; x1<gridLoc[gridInd(x+1,y+1)][0]; ++x1){
					for(auto y1=(int)gridLoc[gridInd(x,y)][1]; y1<gridLoc[gridInd(x+1,y+1)][1]; ++y1){
						Vec3b pix = input.at<Vec3b>(y1,x1);
						pixs[0].push_back((double)pix[0] / 255.0);
						pixs[1].push_back((double)pix[1] / 255.0);
						pixs[2].push_back((double)pix[2] / 255.0);
					}
				}
				Vector3d vars(math_util::variance(pixs[0]),math_util::variance(pixs[1]),math_util::variance(pixs[2]));
				saliency[sid] = vars.norm();

			}
		}
	}

	void GridWarpping::addCoefficient(std::vector<Eigen::Triplet<double> > &triplets, Eigen::VectorXd &B, const int x, const int y, int &cInd, const int condition) const {


					int cgid = gridInd(x, y);
					double wsimilarity;
					Vector2i gid;
					Vector2d refUV;

	      	if (condition == 1){

		          //V3(i-1,j-1)
		            	    //  |
							//  |____
		          //V2   V1(i,j)

		      		gid = Vector2i(gridInd(x-1, y), gridInd(x-1, y-1)); // V2 , V3
							refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
							wsimilarity = alpha_ * saliency[sailencyInd(x-1, y-1)];

	      			} else if (condition == 2) {

	      	     // V3   V2
	      	            	  //         _____
	      					  //                 |
	      	            	  //                 |
	      	     //      V1(i,j)

			      		gid = Vector2i(gridInd(x, y-1), gridInd(x-1, y-1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y-1)];

	      			}  else if (condition == 3) {

			      		gid = Vector2i(gridInd(x, y-1), gridInd(x+1, y-1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x, y-1)];

	      			}  else if (condition == 4) {

			      		gid = Vector2i(gridInd(x+1, y), gridInd(x+1, y-1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x, y-1)];

	      			}  else if (condition == 5) {

			      		gid = Vector2i(gridInd(x+1, y), gridInd(x+1, y+1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x, y)];

	      			}  else if (condition == 6) {

			      		gid = Vector2i(gridInd(x, y+1), gridInd(x+1, y+1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x, y)];

	      			}  else if (condition == 7) {

			      		gid = Vector2i(gridInd(x, y+1), gridInd(x-1, y+1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			}  else if (condition == 8) {

			      		gid = Vector2i(gridInd(x-1, y), gridInd(x-1, y+1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			} else if (condition == 9) {

			      		gid = Vector2i(gridInd(x-1, y), gridInd(x, y-1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			} else if (condition == 10) {

			      		gid = Vector2i(gridInd(x, y-1), gridInd(x+1, y)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			} else if (condition == 11) {

			      		gid = Vector2i(gridInd(x+1, y), gridInd(x, y+1)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			} else if (condition == 12) {

			      		gid = Vector2i(gridInd(x, y+1), gridInd(x-1, y)); // V2 , V3
								refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]); // V1, V2 ,V3
								wsimilarity = alpha_ * saliency[sailencyInd(x-1, y)];

	      			}

	    // V1 -V2 -uV3 +uV2 + (v(V3_y - V2_y), v(V2_x-V3_x)) = 0

	    //cout << refUV[0] << "," << refUV[1] << endl;

	    //x coordinate
			triplets.push_back(Triplet<double>(cInd, cgid * 2, wsimilarity)); // V1
			triplets.push_back(Triplet<double>(cInd, gid[0] * 2, -1 * wsimilarity)); // -V2
			triplets.push_back(Triplet<double>(cInd, gid[0] * 2, refUV[0] * wsimilarity)); // uV2
			triplets.push_back(Triplet<double>(cInd, gid[1] * 2, -1 * refUV[0] * wsimilarity)); // -uV3

			triplets.push_back(Triplet<double>(cInd, gid[0] * 2 + 1, -1 * refUV[1] * wsimilarity)); // -vV2_y
			triplets.push_back(Triplet<double>(cInd, gid[1] * 2 + 1, refUV[1] * wsimilarity)); // vV3_y
			B[cInd] = 0;

			//y coordinate
			triplets.push_back(Triplet<double>(cInd + 1, cgid * 2 + 1, wsimilarity)); // V1
			triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, -1 * wsimilarity));  // - V2
			triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, refUV[0] * wsimilarity)); // uV2
			triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2 + 1, -1 * refUV[0] * wsimilarity)); // -uV3
			triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2, refUV[1] * wsimilarity)); // vV2_x
			triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2, -1 * refUV[1] * wsimilarity)); // -vV3_x

			B[cInd + 1] = 0;
			cInd += 2;


		  }


	void GridWarpping::warpImageCloseForm(const cv::Mat &input, cv::Mat &output, const std::vector<Eigen::Vector2d>& pts1, const std::vector<Eigen::Vector2d>& pts2) const{
		//CHECK_EQ(pts1.size(), pts2.size());

		char buffer[1024] = {};

		vector<Vector2d> resGrid(gridLoc.size());
		const int kDataTerm = (int)pts1.size() * 2;
		const int kSimTerm = (gridW-1)*(gridH-1)*2*8 + (gridH +1 + gridW +1 -4)*2*4*2 + 4*2*2; //total number of row
		//const int kSimTerm = (gridW-1)*(gridH-1)*8;
		const int kVar = (int)gridLoc.size() * 2;

		vector<Eigen::Triplet<double> > triplets;
		VectorXd B(kDataTerm+kSimTerm);
		//add data constraint


		int cInd = 0;
		//cout << pts2.size() << endl;

		for(auto i=0; i<pts1.size(); ++i) {

			if (pts1[i][0] < 0 || pts1[i][1] < 0 || pts1[i][0] >= width - 1 || pts1[i][1] >= height - 1)
				continue;
			Vector4i indRef;
			Vector4d bwRef;
			//CHECK_LT(cInd + 1, B.rows());

			// 원래 구현이랑 다른것을 주의
			// 원래 구현된건.. inverse mapping을 위해  target -> input 으로의 warped mesh를 계산했음.
			getGridIndAndWeight(pts1[i], indRef, bwRef);
			//getGridIndAndWeight(pts2[i], indRef, bwRef);
			for (auto j = 0; j < 4; ++j) {
				//CHECK_LT(indRef[j]*2+1, kVar);
				triplets.push_back(Triplet<double>(cInd, indRef[j] * 2, wdata_ * bwRef[j]));
				triplets.push_back(Triplet<double>(cInd + 1, indRef[j] * 2 + 1, wdata_ * bwRef[j]));
			}

			// 원래 구현이랑 target이 반대인것에 주의.
			// 원래 구현된건.. inverse mapping을 위해  target -> input 으로의 warped mesh를 계산했음.

			//B[cInd] = wdata_ * pts1[i][0];
			//B[cInd + 1] = wdata_ * pts1[i][1];
			B[cInd] = wdata_ * pts2[i][0];
			B[cInd + 1] = wdata_ * pts2[i][1];
			cInd += 2;
		}

		//cout << "complete add data term" << endl;

		// y=0, x=0
		addCoefficient(triplets, B, 0, 0, cInd, 5);
		addCoefficient(triplets, B, 0, 0, cInd, 6);

		// y=0, x=gridW
		addCoefficient(triplets, B, gridW, 0, cInd, 7);
		addCoefficient(triplets, B, gridW, 0, cInd, 8);

		// y=gridH, x=0
		addCoefficient(triplets, B, 0, gridH, cInd, 3);
		addCoefficient(triplets, B, 0, gridH, cInd, 4);

		// y=gridH, x=gridW
		addCoefficient(triplets, B, gridW, gridH, cInd, 1);
		addCoefficient(triplets, B, gridW, gridH, cInd, 2);

			for (auto x = 1; x <= gridW-1; ++x) {

				addCoefficient(triplets, B, x, 0, cInd, 5);
				addCoefficient(triplets, B, x, 0, cInd, 6);
				addCoefficient(triplets, B, x, 0, cInd, 7);
				addCoefficient(triplets, B, x, 0, cInd, 8);

			}

			for (auto x = 1; x <= gridW-1; ++x) {

				addCoefficient(triplets, B, x, gridH, cInd, 1);
				addCoefficient(triplets, B, x, gridH, cInd, 2);
				addCoefficient(triplets, B, x, gridH, cInd, 3);
				addCoefficient(triplets, B, x, gridH, cInd, 4);
			}

			for (auto y = 1; y <= gridH-1; ++y) {

				addCoefficient(triplets, B, 0, y, cInd, 3);
				addCoefficient(triplets, B, 0, y, cInd, 4);
				addCoefficient(triplets, B, 0, y, cInd, 5);
				addCoefficient(triplets, B, 0, y, cInd, 6);

			}

			for (auto y = 1; y <= gridH-1; ++y) {

				addCoefficient(triplets, B, gridW, y, cInd, 1);
				addCoefficient(triplets, B, gridW, y, cInd, 2);
				addCoefficient(triplets, B, gridW, y, cInd, 7);
				addCoefficient(triplets, B, gridW, y, cInd, 8);

			}

		for(auto y=1; y<= gridH-1; ++y) {
			for (auto x = 1; x <= gridW-1; ++x) {
				for (int i=1; i<=8; ++i){
					addCoefficient(triplets, B, x, y, cInd, i);
				}
				//for (int i=9; i<=12; ++i){
				//	addCoefficient(triplets, B, x, y, cInd, i);
				//}

			}
		}

		/*
		double wsimilarity = 1.0;
		for(auto y=1; y< gridH; ++y) {
			for (auto x = 1; x < gridW; ++x) {
				vector<Vector2i> gids{
						Vector2i(gridInd(x - 1, y), gridInd(x, y - 1)),
						Vector2i(gridInd(x, y - 1), gridInd(x + 1, y)),
						Vector2i(gridInd(x + 1, y), gridInd(x, y + 1)),
						Vector2i(gridInd(x, y + 1), gridInd(x - 1, y))
				};
				const int cgid = gridInd(x, y);
//				printf("-----------------------\n");
				for (const auto &gid: gids) {
					Vector2d refUV = getLocalCoord(gridLoc[cgid], gridLoc[gid[0]], gridLoc[gid[1]]);
//					printf("(%.2f,%.2f),(%.2f,%.2f),(%.2f,%.2f), u:%.2f,v:%.2f\n", gridLoc[cgid][0], gridLoc[cgid][1],
//						   gridLoc[gid[0]][0], gridLoc[gid[0]][1], gridLoc[gid[1]][0], gridLoc[gid[1]][1], refUV[0],
//						   refUV[1]);


					// V1 -V2 -uV3 +uV2 + (v(V3_y - V2_y), -v(V3_x - V2_x)) 가 된다.
					// 같은 인덱스는 setFromTriplets에서 다 summation 되버린다.
					// 논문과는 다르게 하나의 vertex에 대해서 4개의 삼각형만 사용하는 것 같고 boundary에 대해서는 계산 안하는것 같다.


					//x coordinate
					triplets.push_back(Triplet<double>(cInd, cgid * 2, wsimilarity)); // V1
					triplets.push_back(Triplet<double>(cInd, gid[0] * 2, -1 * wsimilarity)); // -V2
					triplets.push_back(Triplet<double>(cInd, gid[0] * 2, refUV[0] * wsimilarity)); // uV2
					triplets.push_back(Triplet<double>(cInd, gid[1] * 2, -1 * refUV[0] * wsimilarity)); // -uV3

					triplets.push_back(Triplet<double>(cInd, gid[0] * 2 + 1, -1 * refUV[1] * wsimilarity)); // -vV2_y
					triplets.push_back(Triplet<double>(cInd, gid[1] * 2 + 1, refUV[1] * wsimilarity)); // vV3_y
					B[cInd] = 0;

					//y coordinate
					triplets.push_back(Triplet<double>(cInd + 1, cgid * 2 + 1, wsimilarity)); // V1
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, -1 * wsimilarity));  // - V2
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2 + 1, refUV[0] * wsimilarity)); // uV2
					triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2 + 1, -1 * refUV[0] * wsimilarity)); // -uV3
					triplets.push_back(Triplet<double>(cInd + 1, gid[0] * 2, refUV[1] * wsimilarity)); // vV2_x
					triplets.push_back(Triplet<double>(cInd + 1, gid[1] * 2, -1 * refUV[1] * wsimilarity)); // -vV3_x

					B[cInd + 1] = 0;
					cInd += 2;

				}
			}
		}
		*/
		//cout << "complete add sim term" << endl;
//		const double wregular = 0.1;
//		for(auto x=0; x<=gridW; ++x){
//			for(auto y=0; y<=gridH; ++y){
//				int gid = gridInd(x,y);
//				triplets.push_back(Triplet<double>(cInd, gid*2, wregular));
//				triplets.push_back(Triplet<double>(cInd+1, gid*2+1, wregular));
//				B[cInd] = wregular * gridLoc[gid][0];
//				B[cInd+1] = wregular * gridLoc[gid][1];
//				cInd +=2;
//			}
//		}
		//CHECK_LE(cInd, kDataTerm+kSimTerm);
		SparseMatrix<double> A(cInd, kVar);
		A.setFromTriplets(triplets.begin(), triplets.end());
		//cout << "complete make sparse matrix term" << endl;



    double start = (double)getTickCount();

    //A.makeCompressed();
    //Eigen::SparseQR<SparseMatrix<double>, COLAMDOrdering<int> > solver(A);
    //VectorXd res = solver.solve(B.block(0,0,cInd,1));

		Eigen::SPQR<SparseMatrix<double> > solver(A);
		VectorXd res = solver.solve(B.block(0,0,cInd,1));
    double timeSec = (getTickCount() - start) / getTickFrequency();
    cout << "complete solve : " << timeSec << " sec" << endl;

		//CHECK_EQ(res.rows(), kVar);

		vector<Vector2d> vars(gridLoc.size());
		for(auto i=0; i<vars.size(); ++i){
			vars[i][0] = res[2*i];
			vars[i][1] = res[2*i+1];
		}


		output = Mat(height, width, CV_32FC2, Scalar::all(0));
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vector4i ind;
				Vector4d w;
				getGridIndAndWeight(Vector2d(x,y), ind, w);
				Vector2d pt(0,0);
				for(auto i=0; i<4; ++i){
					pt[0] += vars[ind[i]][0] * w[i];
					pt[1] += vars[ind[i]][1] * w[i];
				}
				//if(pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
				//					continue;

				//Vector2d pixO = interpolation_util::bilinear<float,2>((float*)input.data, input.cols, input.rows, pt);
				//output.at<Point2f>(y,x) = Point2f((float)pixO[0], (float)pixO[1]);

				output.at<Point2f>(y,x) = Point2f((float)(pt[0]-x), (float)(pt[1]-y));
			}
		}

		/*
		output = Mat(height, width, CV_32FC2, Scalar::all(0));
		for(auto y=0; y<height; ++y){
			for(auto x=0; x<width; ++x){
				Vector4i ind;
				Vector4d w;
				getGridIndAndWeight(Vector2d(x,y), ind, w);
				Vector2d pt(0,0);
				for(auto i=0; i<4; ++i){
					pt[0] += vars[ind[i]][0] * w[i];
					pt[1] += vars[ind[i]][1] * w[i];
				}
				if(pt[0] < 0 || pt[1] < 0 || pt[0] > width - 1 || pt[1] > height - 1)
									continue;

				Vector2d pixO = interpolation_util::bilinear<float,2>((float*)input.data, input.cols, input.rows, pt);
				//output.at<Point2f>(y,x) = Point2f((float)pixO[0], (float)pixO[1]);

				output.at<Point2f>(y,x) = Point2f((float)((pt[0]+pixO[0])-x), (float)((pt[1]+pixO[1])-y));
				cout <<"pt : "<< pt << " pixO : " <<pixO << " output : "<<output.at<Point2f>(y,x) <<endl;
				cout << x<< y<< endl;
			}
		}
		*/
		/*
		// visualize grid
		Mat inputOutput = Mat::zeros(input.rows, input.cols, CV_8UC3);
		visualizeGrid(gridLoc, inputOutput);
		//for(const auto& pt: pts1)
		 	//cv::circle(inputOutput, cv::Point2d(pt[0], pt[1]), 1, Scalar(0,0,255), 2);
		  //sprintf(buffer, "vis_input%05d.jpg", id);

		imshow("pre_wpared_grid", inputOutput);
		waitKey(10);


		Mat outputOutput = Mat::zeros(input.rows, input.cols, CV_8UC3);
		visualizeGrid(vars, outputOutput);
		//for(const auto& pt: pts2)
		 	//cv::circle(outputOutput, cv::Point2d(pt[0], pt[1]), 1, Scalar(0,0,255), 2);
		  //sprintf(buffer, "vis_input%05d.jpg", id);
		//cout << outputOutput.size() << endl;
		imshow("warpped_grid", outputOutput);
		waitKey(10);
		*/
	}

}//namespace substablas

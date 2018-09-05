/*
 * SttedyFlow.cpp
 *
 *  Created on: 2018. 8. 21.
 *      Author: rjs
 */

#include "SteedyFlow.h"
#include "warping.h"
#include "utility.h"

using namespace std;
using namespace cv;
using namespace visualization_util;

SteedyFlow::SteedyFlow() {

	// spatial analysis
	setThreshold(0.1);

	// motion completion
	setW_data(1.0);
	setW_s(1.0);
	setComputeSaliency(true);
	setUseMaskBoundary(true);

	// compute t_window
	setTau(20);
	setMinWindow(1);

	// smooth motion
	setLambda(2.5);
	setSmoothIter(10);
	setUpdateWindow(true);

	// temporal analysis
	setAlpha(20);
	setEpsilon(0.2);
	setKernelSize(9);
	setKernelStd(3.0);

	setIter(1);
}

/*
void SteedyFlow::spatialAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	for (int t=0; t<C.size(); ++t){

	  vector<Mat> channels(2);
	  // split img:
	  split(C[t], channels);
	  // get the channels (dont forget they follow BGR order in OpenCV)

	  Mat magnitudeFlow = cv::abs(channels[0]) + cv::abs(channels[1]);

	  Mat sobelx, sobely;
		Sobel(magnitudeFlow, sobelx, CV_32F, 1, 0, 1);
		Sobel(magnitudeFlow, sobely, CV_32F, 0, 1, 1);


		int rows = C[t].rows;
		int cols = C[t].cols;
		Mat mask_t(rows, cols, CV_8UC1, Scalar::all(255));
    //spatial-analysis
		double threshold_pow = threshold_*threshold_;
    for (int y=0; y< rows; ++y){
    	for (int x=0; x< cols; ++x){
    		double gd_magnitude = pow((double)sobelx.at<float>(y,x),2) + pow((double)sobely.at<float>(y,x),2);
    		if(gd_magnitude > threshold_pow)
    			mask_t.at<uchar>(y,x) = 0;
    		}
    	}

    mask[t] = mask_t.clone();
    //Mat dst;
    //cv::dilate(mask_t, dst, cv::Mat());
    //imshow("spatial_mask2", mask_t);
    //imshow("dailated", mask_t);
    //waitKey(100);
	}
}
*/


void SteedyFlow::spatialAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	for (int t=0; t<(int)C.size(); ++t){
		Mat ch1, ch2;

	  vector<Mat> channels(2);
	  // split img:
	  split(C[t], channels);
	  // get the channels (dont forget they follow BGR order in OpenCV)
	  ch1 = channels[0]; //flowx
	  ch2 = channels[1]; //flowy

	  Mat ch1_sobelx, ch1_sobely;
		Sobel(ch1, ch1_sobelx, CV_32F, 1, 0, 1);
		Sobel(ch1, ch1_sobely, CV_32F, 0, 1, 1);

		Mat ch2_sobelx, ch2_sobely;
		Sobel(ch2, ch2_sobelx, CV_32F, 1, 0, 1);
		Sobel(ch2, ch2_sobely, CV_32F, 0, 1, 1);

		int rows = C[t].rows;
		int cols = C[t].cols;
		Mat mask_t(rows, cols, CV_8UC1, Scalar::all(255));
    //spatial-analysis

		double threshold_pow = threshold_*threshold_;
    for (int y=0; y< rows; ++y){
    	for (int x=0; x< cols; ++x){
    		double gd_x = pow((double)ch1_sobelx.at<float>(y,x),2) + pow((double)ch2_sobelx.at<float>(y,x),2);
    		double gd_y = pow((double)ch1_sobely.at<float>(y,x),2) + pow((double)ch2_sobely.at<float>(y,x),2);

    		double gd_magnitude_pow = gd_x + gd_y;

    		if(gd_magnitude_pow > threshold_pow)
    			mask_t.at<uchar>(y,x) = 0;
    		}
    	}

    mask[t]=mask_t.clone();
    //imshow("spatial_mask", mask_t);
    //waitKey(100);
	}
}


/*
void SteedyFlow::completeMotion(const std::vector<cv::Mat> &flows, std::vector<cv::Mat> &completedFlows, const std::vector<cv::Mat> &mask, const std::vector<cv::Mat> &images) const {

	for(size_t t=0; t<(size_t)flows.size(); t++){

		Mat flow_t = flows[t];
		Mat mask_t = mask[t];


		Mat element = getStructuringElement(0, Size(3,3));


		Mat closing_mask;
		cv::morphologyEx(mask_t, closing_mask, MORPH_CLOSE, element);
		//imshow("closing_mask",closing_mask);
		//waitKey(10);

		Mat opening_mask;
		cv::morphologyEx(mask_t, opening_mask, MORPH_OPEN, element);
		//imshow("opening_mask",opening_mask);
		//waitKey(10);
		Mat erod_mask_t;
		cv::erode(opening_mask, erod_mask_t, element);
		//imshow("erod_mask_t",erod_mask_t);
		//waitKey(10);
		Mat control_mask = opening_mask-erod_mask_t;
		//imshow("control_mask",control_mask);
		//waitKey(0);


		Mat erod_mask_t;
	  cv::erode(mask_t, erod_mask_t, cv::Mat());
	  Mat control_mask = mask_t-erod_mask_t;

		//Mat control_mask = mask_t;

		float* p_flow_t = (float*)flow_t.data;
		uchar* p_control_mask = (uchar*)control_mask.data;

		vector<Eigen::Vector2d> pts1, pts2;
		for (int y = 0; y < flow_t.rows; ++y){
			for (int x = 0; x < flow_t.cols; ++x){
				int cInd = y*flow_t.cols+x;
				if (p_control_mask[cInd]==255){
	    			pts1.push_back(Eigen::Vector2d(x, y));
	    			pts2.push_back(Eigen::Vector2d(p_flow_t[cInd*2] + x, p_flow_t[cInd*2+1] + y));
				}
			}
		}

		if (pts2.size() == 0){

			completedFlows[t] = flow_t.clone();
			continue;
		}

		Mat warped;
		substab::GridWarpping warping(flow_t.cols, flow_t.rows, 40, 40);
		warping.setAlpha(w_s_);
		warping.setWdata(w_data_);

		if (computeSaliency_)
			warping.computeSaliencyWeight(images[t]);

		cout << "complete init warping" << endl;
		warping.warpImageCloseForm(flow_t, warped, pts1, pts2);

		double start = (double)getTickCount();
		flann::KDTreeIndexParams indexParams;

		vector<Point2i> refIndex;
		vector<Point2f> indexPoint;
		vector<Point2i> refQuery;
		vector<Point2f> queryPoint;
		for (int y = 0; y < mask_t.rows; ++y){
			for (int x = 0; x < mask_t.cols; ++x){
				if (control_mask.at<uchar>(y,x)==255){
					Point2f warpedPoint = warped.at<Point2f>(y,x);
					refIndex.push_back(Point2i(x,y));
					indexPoint.push_back(Point2f(x+warpedPoint.x, y+warpedPoint.y));
				} else if (mask_t.at<uchar>(y,x)==0) {
				//} else if (opening_mask.at<uchar>(y,x)==0) {
					Point2f warpedPoint = warped.at<Point2f>(y,x);
					refQuery.push_back(Point2i(x,y));
					queryPoint.push_back(Point2f(x+warpedPoint.x, y+warpedPoint.y));
				}
			}
		}


		flann::Index kdtree(Mat(indexPoint).reshape(1), indexParams);
		Mat indices;
		Mat dists;
		kdtree.knnSearch(Mat(queryPoint).reshape(1), indices, dists, 2);


		Mat completedFlow_t = flow_t.clone();
		for (int i=0; i<indices.rows; ++i){
			int index_p1 = indices.at<int>(i,0);
			int index_p2 = indices.at<int>(i,1);
			float dist_p1 = dists.at<float>(i,0);
			float dist_p2 = dists.at<float>(i,1);

			Point2f flow_p1 = flow_t.at<Point2f>(refIndex[index_p1].y, refIndex[index_p1].x);
			Point2f flow_p2 = flow_t.at<Point2f>(refIndex[index_p2].y, refIndex[index_p2].x);
			Point2f interpolatedFlow = (dist_p2/(dist_p1+dist_p2))*flow_p1 + (dist_p1/(dist_p1+dist_p2))*flow_p2;

			Point2i q_point = refQuery[i];
			completedFlow_t.at<Point2f>(q_point.y, q_point.x) = interpolatedFlow;
		}

		completedFlows[t] = completedFlow_t.clone();

	  double timeSec = (getTickCount() - start) / getTickFrequency();
	  cout << "interpolation complete : " << timeSec << " sec" << endl;


	  Mat dst;
		drawOpticalFlow(flow_t, dst);
		imshow("ori_flow",dst);
		waitKey(10);
		drawOpticalFlow(completedFlow_t, dst);
		imshow("completedflow",dst);
		waitKey(10);
		drawOpticalFlow(warped, dst);
		imshow("warped",dst);
		waitKey(10);
		imshow("mask_t",mask_t);
		waitKey(10);
		imshow("controlmask_t",control_mask);
		waitKey(10);
		//imshow("erod_maks", erod_mask_t);
		//waitKey(10);
		//imshow("opening_mask",opening_mask);
		//waitKey(10);


		vector<Point2f> refIndex2;
		vector<Point2f> indexPoint2;
		vector<Point2f> refQuery2;
		vector<Point2f> queryPoint2;
		for (int y = 0; y < mask_t.rows; ++y){
			for (int x = 0; x < mask_t.cols; ++x){
				if (control_mask.at<uchar>(y,x)==255){
					Point2f warpedPoint = warped.at<Point2f>(y,x);
					refIndex2.push_back(Point2f(x,y));
					indexPoint2.push_back(Point2f(x+warpedPoint.x, y+warpedPoint.y));
				} else if (mask_t.at<uchar>(y,x)==0) {
				//} else if (opening_mask.at<uchar>(y,x)==0) {
					Point2f warpedPoint = warped.at<Point2f>(y,x);
					refQuery2.push_back(Point2f(x,y));
					queryPoint2.push_back(Point2f(x+warpedPoint.x, y+warpedPoint.y));
				}
			}
		}

		flann::Index kdtree2(Mat(refIndex2).reshape(1), indexParams);
		Mat indices2;
		Mat dists2;
		kdtree2.knnSearch(Mat(refQuery2).reshape(1), indices2, dists2, 2);


		Mat TEST = flow_t.clone();
		for (int i=0; i<indices2.rows; ++i){
			int index_p1 = indices2.at<int>(i,0);
			int index_p2 = indices2.at<int>(i,1);
			float dist_p1 = dists2.at<float>(i,0);
			float dist_p2 = dists2.at<float>(i,1);

			Point2f flow_p1 = flow_t.at<Point2f>(refIndex[index_p1].y, refIndex[index_p1].x);
			Point2f flow_p2 = flow_t.at<Point2f>(refIndex[index_p2].y, refIndex[index_p2].x);
			Point2f interpolatedFlow = (dist_p2/(dist_p1+dist_p2))*flow_p1 + (dist_p1/(dist_p1+dist_p2))*flow_p2;

			Point2i q_point = refQuery[i];
			TEST.at<Point2f>(q_point.y, q_point.x) = interpolatedFlow;
		}

		drawOpticalFlow(TEST, dst);
		imshow("TEST",dst);
		waitKey(10);

	}

	cv::destroyAllWindows();
}
*/

void SteedyFlow::completeMotion(const std::vector<cv::Mat> &flows, std::vector<cv::Mat> &completedFlows, const std::vector<cv::Mat> &mask, const std::vector<cv::Mat> &images) const {

	for(size_t t=0; t<(size_t)flows.size(); t++){

		Mat flow_t = flows[t];
		Mat mask_t = mask[t];


	  //Mat dil_mask_t;
	  //cv::dilate(mask_t, dil_mask_t, cv::Mat());
	  //Mat control_mask = dil_mask_t - mask_t;
		Mat control_mask;
		if (useMaskBoundary_){
			Mat erod_mask_t;
		  cv::erode(mask_t, erod_mask_t, cv::Mat());
		  control_mask = mask_t-erod_mask_t;
		} else {
			control_mask = mask_t;
		}

		float* p_flow_t = (float*)flow_t.data;
		uchar* p_control_mask = (uchar*)control_mask.data;
		uchar* p_mask_t = (uchar*)mask_t.data;

		vector<Eigen::Vector2d> pts1, pts2;
		vector<bool> nonZero;
		for (int y = 0; y < flow_t.rows; ++y){
			for (int x = 0; x < flow_t.cols; ++x){
				int cInd = y*flow_t.cols+x;

				//find nonzero elements on control mask
				if (p_control_mask[cInd]==255){
	    			pts1.push_back(Eigen::Vector2d(x, y));
	    			pts2.push_back(Eigen::Vector2d(p_flow_t[cInd*2] + x, p_flow_t[cInd*2+1] + y));
				}
				//find nonzero elements on original mask
				if (p_mask_t[cInd]==255){
					nonZero.push_back(true);
				}
			}
		}

		// if too small inlier
		if (nonZero.size()<((int)flow_t.rows*(int)flow_t.cols)/2.0 || pts2.size()==0){
		//if (pts2.size()==0){
			completedFlows[t] = flow_t.clone();
			continue;
		}

		Mat warped;
		Warping::GridWarpping warping(flow_t.cols, flow_t.rows, 40, 40);
		warping.setAlpha(w_s_);
		warping.setWdata(w_data_);

		if (computeSaliency_)
			warping.computeSaliencyWeight(images[t]);

		cout << "complete init warping" << endl;
		warping.warpImageCloseForm(flow_t, warped, pts1, pts2);


		if (useMaskBoundary_){
			Mat completedFlow_t = flow_t.clone();
			float* p_completedFlow_t = (float*)completedFlow_t.data;
			float* p_warped_t = (float*)warped.data;
			//uchar* p_mask_t = (uchar*)mask_t.data;

			for (int y = 0; y < flow_t.rows; ++y){
				for (int x = 0; x < flow_t.cols; ++x){
					int cInd = y*flow_t.cols+x;
					if (p_mask_t[cInd]!=255){
						p_completedFlow_t[cInd*2] = p_warped_t[cInd*2];
						p_completedFlow_t[cInd*2+1] = p_warped_t[cInd*2+1];
					}
				}
			}
			completedFlows[t] = completedFlow_t.clone();
		} else {
			completedFlows[t] = warped.clone();
		}

		Mat dst;
		drawOpticalFlow(flow_t, dst);
		imshow("ori_flow",dst);
		waitKey(10);
		drawOpticalFlow(completedFlows[t], dst);
		imshow("completedflow",dst);
		waitKey(10);
		imshow("mask_t",mask_t);
		waitKey(10);


	}

	cv::destroyAllWindows();
}

void SteedyFlow::computeTemporalWindow(const std::vector<cv::Mat> &flows, std::vector<cv::Point2i> &t_Windows) const {


	for(int t=0; t<(int)flows.size(); t++){

		//forward tracing
		Mat forward_trace = flows[t].clone();
		int forward_wt;
		int i = 0;
		for(;;){
			if (t+i >= (int)flows.size()-1){
				forward_wt = i;
				break;
			}

			// check forward_trace
		  vector<Mat> channels(2);
		  split(forward_trace, channels);
		  Mat l1norm_image;
		  l1norm_image = cv::abs(channels[0]) + cv::abs(channels[1]);

			double maxVal;
			//cout<<maxVal<<endl;
			minMaxLoc(l1norm_image, 0, &maxVal);

			if (maxVal > tau_*2+1){
				//cout << t << "frame's " << "forward w_t : "<< i << " maxVal : " << maxVal<< endl;
				forward_wt=i;

				break;
			}

			i++;

			// update forward_trace
		  Mat map(forward_trace.size(), CV_32FC2);
		  float* p_forward_trace = (float*)forward_trace.data;
		  float* p_map = (float*)map.data;
		  for (int y = 0; y < map.rows; ++y){
			  for (int x = 0; x < map.cols; ++x){
				  p_map[(y*map.cols+x)*2] = p_forward_trace[(y*map.cols+x)*2] + x;
				  p_map[(y*map.cols+x)*2+1] = p_forward_trace[(y*map.cols+x)*2+1] + y;
			  }
		  }
			Mat ith_flow = flows.at(i+t);
			Mat out;
		  //cv::remap(ith_flow, out, map, cv::Mat(), CV_INTER_LINEAR);
		  cv::remap(ith_flow, out, map, cv::Mat(), CV_INTER_NN);

		  forward_trace += out;

			}

		// backward tracing
		int backward_wt;
		Mat backward_trace;
		i=1;
		if (t-i >= 0)
			flows[t-i].copyTo(backward_trace);

		for(;;){
			if (t-i < 0){
				backward_wt=i-1;
				break;
			}

			// check backward_trace
		  vector<Mat> channels(2);
		  split(backward_trace, channels);
		  Mat l1norm_image;
		  l1norm_image = cv::abs(channels[0]) + cv::abs(channels[1]);

			double maxVal;
			//cout<<maxVal<<endl;
			minMaxLoc(l1norm_image, 0, &maxVal);

			if (maxVal > tau_*2+1){
				//cout << t << "frame's " << "backward w_t : "<< i-1 << " maxVal : " << maxVal<< endl;
				backward_wt=i-1;
				break;
			}

			i++;

			if (t-i < 0){
				forward_wt = i-1;
				break;
			}
			// update forward_trace
			Mat ith_flow = flows[t-i];
		  Mat map(backward_trace.size(), CV_32FC2);
		  float* p_ith_flow = (float*)ith_flow.data;
		  float* p_map = (float*)map.data;
		  for (int y = 0; y < map.rows; ++y){
			  for (int x = 0; x < map.cols; ++x){
				  p_map[(y*map.cols+x)*2] = p_ith_flow[(y*map.cols+x)*2] + x;
				  p_map[(y*map.cols+x)*2+1] = p_ith_flow[(y*map.cols+x)*2+1] + y;
			  }
		  }

			Mat out;
		  //cv::remap(backward_trace, out, map, cv::Mat(), CV_INTER_LINEAR);
		  cv::remap(backward_trace, out, map, cv::Mat(), CV_INTER_NN);

		  backward_trace = ith_flow + out;
		}

		cout << t << " frame's " << "t_window : ("<< backward_wt << " , " << forward_wt << ")"<< endl;
		// check valid forward, backward w_t
		if (forward_wt < minWindow_){
			if (t+minWindow_ >= (int)flows.size()-1)
				forward_wt = (int)flows.size()-1-t;
			else
				forward_wt = minWindow_;
		}

		if (backward_wt < minWindow_){
			if (t-minWindow_ < 0)
				backward_wt = t;
			else
				backward_wt = minWindow_;
		}

		t_Windows.push_back(Point2i(-1*backward_wt, forward_wt));

		}
}

void SteedyFlow::computeSimpleTemporalWindow(const std::vector<cv::Mat> &P, std::vector<cv::Point2i> &t_Windows) const {


	for(int t=0; t<(int)P.size(); t++){

		//forward tracing
		Mat forward_trace;
		int forward_wt;
		int i = 0;

		if (t+i > 0)
			forward_trace = P[t+i]-P[t];
		else if (t+i == 0)
			forward_trace = P[t+i];

		for(;;){
			if (t+i >= (int)P.size()-1){
				forward_wt = i;
				break;
			}

			// check forward_trace
		  vector<Mat> channels(2);
		  split(forward_trace, channels);
		  Mat l1norm_image;
		  l1norm_image = cv::abs(channels[0]) + cv::abs(channels[1]);

			double maxVal;
			minMaxLoc(l1norm_image, 0, &maxVal);

			if (maxVal > tau_*2+1){
				//cout << t << "frame's " << "forward w_t : "<< i << " maxVal : " << maxVal<< endl;
				forward_wt=i;

				break;
			}

			i++;

		  forward_trace = P[t+i]-P[t];

			}

		// backward tracing
		int backward_wt;
		Mat backward_trace;
		i=1;
		if (t-i >= 0)
			backward_trace = P[t]-P[t-i];

		for(;;){
			if (t-i < 0){
				backward_wt=i-1;
				break;
			}

			// check backward_trace
		  vector<Mat> channels(2);
		  split(backward_trace, channels);
		  Mat l1norm_image;
		  l1norm_image = cv::abs(channels[0]) + cv::abs(channels[1]);

			double maxVal;
			minMaxLoc(l1norm_image, 0, &maxVal);

			if (maxVal > tau_*2+1){
				//cout << t << "frame's " << "backward w_t : "<< i-1 << " maxVal : " << maxVal<< endl;
				backward_wt=i-1;
				break;
			}

			i++;

			if (t-i < 0){
				forward_wt = i-1;
				break;
			}

		  backward_trace = P[t]-P[t-i];
		}

		// check valid forward, backward w_t

		if (backward_wt<=0 && forward_wt<=0){
			backward_wt=1;
			forward_wt=1;
		}
		//cout << t << " frame's " << "t_window : ("<< backward_wt << " , " << forward_wt << ")"<< endl;

		t_Windows.push_back(Point2i(-1*backward_wt, forward_wt));

		}
}

void SteedyFlow::smoothFlow(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &P, std::vector<cv::Point2i> &t_Windows) const{

	for (int i=0; i<smoothIter_; ++i){
		cout << i << "th iteration start" <<endl;
		double start = (double)getTickCount();

		if (updateWindow_){
			vector<Mat> S;
			S.push_back(P[0].clone());
			for (int t=1; t<(int)P.size(); ++t){
				Mat S_t = P[t]-P[t-1];
				S.push_back(S_t);
			}

			vector<Point2i> t_Windows;
			computeTemporalWindow(S, t_Windows);
			//computeSimpleTemporalWindow(P, t_Windows);
		}

		vector<Mat> result(C.size());
		for (int t=0; t<(int)C.size(); ++t){

			Point2i t_w = t_Windows[t];
			float sigma = (float)(-1*t_w.x + t_w.y)/3.0;

			Mat Ct = C[t];
			Mat sum_Pt(Ct.size(), CV_32FC2, Scalar::all(0));
			float sum_w = 0.0;

			for (int r=t_w.x; r<=t_w.y; ++r){
				if (r==0)
					continue;
				float w_r = exp(-1*r*r/(sigma*sigma));
				sum_Pt += (lambda_*w_r*P[t+r]);
				sum_w += w_r;

			}

			float gamma = 1/(1 + lambda_ * sum_w);
			result[t] = gamma*(Ct + sum_Pt);
		}

		//cout << P[0]<< endl;
		/*
		for (int t=0; t<P.size(); ++t){

				vector<Mat> channels(2);
				Mat sub = P[t]-result[t];
			  split(sub, channels);
			  Mat l1norm_image;
			  l1norm_image = cv::abs(channels[0]) + cv::abs(channels[1]);

				double maxVal;
				minMaxLoc(l1norm_image, 0, &maxVal);
				cout<<"pt-result maxVal"<<maxVal<<endl;

		}
		*/
		for (int t=0; t<(int)C.size(); ++t){
			P[t] = result[t].clone();
		}

		//vector<Mat> mask(C.size());
		//temporalAnalysis(P, mask);

		double timeSec = (getTickCount() - start) / getTickFrequency();
		cout << i << "th iteration complete : " <<timeSec << "sec" <<endl;
	}
}

/*
void SteedyFlow::temporalAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	double threshold = (1 + pow(alpha_, 1/(float)iter_))*epsilon_;

	for (size_t t=0; t<C.size(); ++t){

		Mat C_t = C[t];
		Mat GC_t;
		cv::GaussianBlur(C_t, GC_t, Size(kernelsize_, kernelsize_), kernelstd_, kernelstd_);

		float* p_C_t = (float*)C_t.data;
		float* p_GC_t = (float*)GC_t.data;

		Mat mask_t = Mat(C_t.size(), CV_8UC1, Scalar::all(255));
		uchar* p_mask_t = mask_t.data;

		for (int y=0; y<C_t.rows; ++y){
			for (int x=0; x<C_t.cols; ++x){

				int cInd = (y*C_t.cols+x)*2;
				double sub_x = p_C_t[cInd] - p_GC_t[cInd];
				double sub_y = p_C_t[cInd+1] - p_GC_t[cInd+1];

				if (sub_x*sub_x + sub_y*sub_y > threshold*threshold){
					p_mask_t[y*C_t.cols+x] = 0;
				}
			}
		}

		mask[t] = mask_t.clone();
	}
}
*/

/*
void SteedyFlow::temporalAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	float threshold = (1 + powf(alpha_, 1/(float)iter_))*epsilon_;
	float sigma = 3.0;

	// maybe 2*ceil(2*sigma)+1
	vector<Mat> S;
	S.push_back(C[0].clone());
	for (int t=1; t<(int)C.size(); ++t){
		Mat S_t = C[t]-C[t-1];
		S.push_back(S_t);
	}

	vector<Point2i> t_Windows;
	//computeTemporalWindow(S, t_Windows);
	computeSimpleTemporalWindow(C, t_Windows);

	for (int t=0; t<C.size(); ++t){

		Mat C_t = C[t];
		Mat GC_t(C_t.size(), CV_32FC2, Scalar::all(0));

		Point2i t_w = t_Windows[t];
		for (int r=t_w.x; r<=t_w.y; ++r){
			float w_r = exp(-1*r*r/(sigma*sigma));
			GC_t += (w_r*C[t+r]);
		}

		float* p_C_t = (float*)C_t.data;
		float* p_GC_t = (float*)GC_t.data;

		Mat mask_t = Mat(C_t.size(), CV_8UC1, Scalar::all(0));
		uchar* p_mask_t = mask_t.data;

		for (int y=0; y<C_t.rows; ++y){
			for (int x=0; x<C_t.cols; ++x){
				int cInd = (y*C_t.cols+x)*2;
				float sub_x = p_C_t[cInd] - p_GC_t[cInd];
				float sub_y = p_C_t[cInd+1] - p_GC_t[cInd+1];
				//cout<<sub_x*sub_x<< ", " << sub_y*sub_y<< ", " << threshold*threshold <<endl;

				if (sub_x*sub_x + sub_y*sub_y < threshold*threshold){
					p_mask_t[y*C_t.cols+x] = 255;
				}
			}
		}
		imshow("mask_t", mask_t);
		waitKey(10);
		mask[t] = mask_t.clone();
	}
}
*/


void SteedyFlow::temporalAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	double threshold = (1 + pow(alpha_, 1/(double)iter_))*epsilon_;
	// maybe 2*ceil(2*sigma)+1

	vector<float> gKernel;
	for (int k=-1*(kernelsize_-1)/2; k<=(kernelsize_-1)/2; ++k){
		gKernel.push_back(exp(-1*k*k/9.0));
	}

	for (int t=0; t<(int)C.size(); ++t){

		Mat C_t = C[t];
		Mat GC_t(C_t.size(), CV_32FC2, Scalar::all(0));
		float sum_w = 0.0;
		for (int k=-1*(kernelsize_-1)/2; k<=(kernelsize_-1)/2; ++k){

			int tk = t+k;
			if (tk<0)
				//tk=0;
				continue;

			if (tk>(int)C.size()-1)
				//tk=(int)C.size()-1;
				continue;

			float gWeight = gKernel[k+(kernelsize_-1)/2];

			sum_w += gWeight;
			GC_t += (gWeight * C[tk]);
		}

		GC_t = (1/sum_w)*GC_t;


		float* p_C_t = (float*)C_t.data;
		float* p_GC_t = (float*)GC_t.data;

		Mat mask_t = Mat(C_t.size(), CV_8UC1, Scalar::all(0));
		uchar* p_mask_t = mask_t.data;

		for (int y=0; y<C_t.rows; ++y){
			for (int x=0; x<C_t.cols; ++x){
				int cInd = (y*C_t.cols+x)*2;
				double sub_x = p_C_t[cInd] - p_GC_t[cInd];
				double sub_y = p_C_t[cInd+1] - p_GC_t[cInd+1];
				//cout<<sub_x*sub_x<< ", " << sub_y*sub_y<< ", " << threshold*threshold <<endl;

				//if (abs(sub_x) + abs(sub_y) < threshold){
				if (sub_x*sub_x + sub_y*sub_y < threshold*threshold){
					p_mask_t[y*C_t.cols+x] = 255;
				}
			}
		}

		mask[t] = mask_t.clone();
	}
}



/*
void SteedyFlow::temporalAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const {

	float threshold = (1 + powf(alpha_, 1/(float)iter_))*epsilon_;
	// maybe 2*ceil(2*sigma)+1

	vector<float> gKernel;
	for (int k=-1*(kernelsize_-1)/2; k<=(kernelsize_-1)/2; ++k){
		gKernel.push_back(exp(-1*k*k/9.0));
	}

	for (int t=0; t<C.size(); ++t){
		Mat C_t = C[t];
		Mat GC_t(C_t.size(), CV_32FC2, Scalar::all(0));

		float sum_w=0.0;
		for (int k=-1*(kernelsize_-1)/2; k<=(kernelsize_-1)/2; ++k){
			float gWeight = gKernel[k+(kernelsize_-1)/2];

			int tk = t+k;
			if (tk<0 || tk>C.size()-1)
				continue;

			Mat GC_tk;
			cv::GaussianBlur(C[tk], GC_tk, Size(3, 3), kernelstd_, kernelstd_);

			sum_w += gWeight;
			GC_t += (gWeight * GC_tk);
		}

		GC_t = (1/sum_w)*GC_t;

		float* p_C_t = (float*)C_t.data;
		float* p_GC_t = (float*)GC_t.data;

		Mat mask_t = Mat(C_t.size(), CV_8UC1, Scalar::all(0));
		uchar* p_mask_t = mask_t.data;

		for (int y=0; y<C_t.rows; ++y){
			for (int x=0; x<C_t.cols; ++x){
				int cInd = (y*C_t.cols+x)*2;
				float sub_x = p_C_t[cInd] - p_GC_t[cInd];
				float sub_y = p_C_t[cInd+1] - p_GC_t[cInd+1];
				//cout<< p_C_t[cInd] << ", "<< p_GC_t[cInd] << ", "<< p_C_t[cInd+1] << ", "<< p_GC_t[cInd+1] <<endl;
				//cout<<sub_x*sub_x<< ", " << sub_y*sub_y<< ", " << threshold*threshold <<endl;

				if (sub_x*sub_x + sub_y*sub_y < threshold*threshold){
					p_mask_t[y*C_t.cols+x] = 255;
				}
			}
		}

		imshow("mask_t", mask_t);
		waitKey(10);
		mask[t] = mask_t.clone();
	}
}
*/

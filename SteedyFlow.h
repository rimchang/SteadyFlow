/*
 * SteedyFlow.h
 *
 *  Created on: 2018. 8. 21.
 *      Author: rjs
 */

#ifndef STEEDYFLOW_H_
#define STEEDYFLOW_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

class SteedyFlow {

public:

	SteedyFlow();
	void spatialAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const;

	void completeMotion(const std::vector<cv::Mat> &flows, std::vector<cv::Mat> &completedFlows, const std::vector<cv::Mat> &mask, const std::vector<cv::Mat> &images) const;

	void computeTemporalWindow(const std::vector<cv::Mat> &flows, std::vector<cv::Point2i> &t_Windows) const ;

	void computeSimpleTemporalWindow(const std::vector<cv::Mat> &P, std::vector<cv::Point2i> &t_Windows) const;

	void smoothFlow(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &P, std::vector<cv::Point2i> &t_Windows) const;

	void temporalAnalysis(const std::vector<cv::Mat> &C, std::vector<cv::Mat> &mask) const ;

	void setThreshold(double val) { threshold_ = val; }
	double threshold() const { return threshold_; }

	void setW_s(double val) { w_s_ = val; }
	double w_s() const { return w_s_; }

	void setW_data(double val) { w_data_ = val; }
	double w_data() const { return w_data_; }

	void setComputeSaliency(bool val) { computeSaliency_ = val; }
	bool computeSaliency() const { return computeSaliency_; }

	void setUseMaskBoundary(bool val) { useMaskBoundary_ = val; }
	bool UseMaskBoundary() const { return useMaskBoundary_; }

	void setTau(int val) { tau_ = val; }
	int tau() const { return tau_; }

	void setMinWindow(int val) { minWindow_ = val; }
	int minWindow() const { return minWindow_; }

	void setLambda(float val) { lambda_ = val; }
	float lambda() const { return lambda_; }

	void setSmoothIter(int val) { smoothIter_ = val; }
	int smoothIter() const { return smoothIter_; }

	void setUpdateWindow(bool val) { updateWindow_ = val; }
	bool updateWindow() const { return updateWindow_; }

	void setAlpha(double val) { alpha_ = val; }
	double alpha() const { return alpha_; }

	void setEpsilon(double val) { epsilon_ = val; }
	double epsilon() const { return epsilon_; }

	void setKernelSize(int val) { kernelsize_ = val; }
	int kernelSize() const { return kernelsize_; }

	void setKernelStd(float val) { kernelstd_ = val; }
	float kernelStd() const { return kernelstd_; }

	void setIter(int val) { iter_ = val; }
	int iter() const { return iter_; }
private:
	// spatialAnalysis
	double threshold_;

	// complete motion
	double w_s_;
	double w_data_;
	bool computeSaliency_;
	bool useMaskBoundary_;

	// coumpute TemporalWindow
	int tau_;
	int minWindow_;

	// smoothFlow
	float lambda_;
	int smoothIter_;
	bool updateWindow_;

	// temporalAnalysis
	double alpha_;
	double epsilon_;
	int kernelsize_;
	float kernelstd_;

	int iter_=1;
};


#endif /* STEEDYFLOW_H_ */

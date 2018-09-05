/*
 * KLTFeatureMatch.h
 *
 *  Created on: 2018. 7. 30.
 *      Author: rjs
 */

#ifndef KLTFEATUREMATCH_H_
#define KLTFEATUREMATCH_H_

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

class KLTFeatureMatch {
private:
   double qualityLevel;
   double minDistance;
   int blockSize;
   bool useHarrisDetector;
   double k;
   int maxCorners;

protected:
    cv::Mat image1, image2;
    std::vector<Point2f> goodFeatures;
    std::vector<Point2f> trackedFeatures;
    vector<uchar> status;
    cv::Mat HMatrix;
    void match();


public:
    KLTFeatureMatch();
    void trackfeature(cv::Mat, cv::Mat);
    void detectfeature();
    void ComputeHomography();
    cv::Mat drawfeaturePoints();
    cv::Mat drawfeatureTrack();
    cv::Mat getWarpImage(Mat image);
    cv::Mat getWarpImage();


    inline cv::Mat getHmatrix() const {return HMatrix;};

};


#endif /* KLTFEATUREMATCH_H_ */

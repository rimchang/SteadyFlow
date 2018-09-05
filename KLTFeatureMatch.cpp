#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>

#include "KLTFeatureMatch.h"

#include <iostream>

using namespace std;
using namespace cv;

KLTFeatureMatch::KLTFeatureMatch(){
    qualityLevel = 0.01;
    minDistance = 10;
    blockSize = 3;
    useHarrisDetector = false;
    k = 0.04;
    maxCorners = 500;
}

Mat KLTFeatureMatch::getWarpImage(Mat image){

    Mat warp_src;
    cv::warpPerspective(image, warp_src, HMatrix, image2.size(), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);

    //std::cout <<HMatrix <<std::endl;
    return warp_src;
}

Mat KLTFeatureMatch::getWarpImage(){

    Mat warp_src;
    cv::warpPerspective(image2, warp_src, HMatrix, image2.size(), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);

    //std::cout <<HMatrix <<std::endl;
    return warp_src;
}

void KLTFeatureMatch::ComputeHomography(){


    vector<Point2f> weedGoodFeatures;
    vector<Point2f> weedTrackedFeatures;

    // weed out bad matches
    for(size_t i=0; i < status.size(); i++) {
        if(status[i]) {
        	weedGoodFeatures.push_back(goodFeatures[i]);
        	weedTrackedFeatures.push_back(trackedFeatures[i]);
        }
    }

	HMatrix = findHomography(goodFeatures, trackedFeatures, CV_RANSAC, 1);
}

void KLTFeatureMatch::detectfeature(){
    //cv::Mat src_gray;
    //cvtColor(image1, src_gray, CV_BGR2GRAY);

    cv::goodFeaturesToTrack(image1, goodFeatures, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );
    cout<<goodFeatures.size()<<endl;
}

void KLTFeatureMatch::match(){

    vector<float> err;
    Size winsize = Size(21,21);
    int maxlevel =3;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    double derivlambda=0.5;
    int flags=0;
    cv::calcOpticalFlowPyrLK(image1, image2, goodFeatures, trackedFeatures, status, err, winsize, maxlevel, criteria, derivlambda, flags);
}


Mat KLTFeatureMatch::drawfeaturePoints(){
    Mat src_copy;
    image1.copyTo(src_copy);
    int thickness =2;
    int radius =1;
    std::vector<cv::Point2f> :: const_iterator it = goodFeatures.begin();
    while(it!=goodFeatures.end()){
    cv::circle(src_copy, *it, radius, Scalar(0, 255, 0), thickness);
        ++it;
        }
   return src_copy;
}


Mat KLTFeatureMatch::drawfeatureTrack(){
    Mat src_copy;
    image2.copyTo(src_copy);
    std::vector<Point2f> :: const_iterator itc = goodFeatures.begin();
    std::vector<Point2f> :: const_iterator itf = trackedFeatures.begin();
    while(itc!=goodFeatures.end()){
        circle(src_copy, *itc, 1, Scalar(0, 0, 255), 2, 8, 0);
        circle(src_copy, *itf, 1, Scalar(255, 0, 0), 2, 8, 0);
        cv::line(src_copy, *itc, *itf, Scalar(0,255,0) );
        itc++;
        itf++;
            }
        return src_copy;
}


void KLTFeatureMatch::trackfeature(Mat I1, Mat I2){
    I1.copyTo(image1);
    I2.copyTo(image2);
    detectfeature();
    match();
    ComputeHomography();
}






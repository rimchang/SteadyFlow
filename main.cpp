#include <highgui.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
//#include <iostream>
//#include <stdio.h>
//#include <fstream>
//#include <math.h>
//#include <Eigen/Eigen>
//#include <cv.h>
//#include <opencv2/highgui.hpp>
//#include <opencv2/videostab/inpainting.hpp>
//#include "opencv2/flann/miniflann.hpp"


#include "utility.h"
#include "warping.h"
#include "KLTFeatureMatch.h"
#include "SteedyFlow.h"
#include "opticalflow/OpticalFlow.h"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace visualization_util;
using namespace vectorMat_util;


int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, "{help h || show help message}"
            "{ @input_path | | input video path}{ @output_path | | output video path}{ @flows_path | | initial optical flow path}");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string input_path = parser.get<string>("@input_path");
    string output_path = parser.get<string>("@output_path");
    string flows_path = parser.get<string>("@flows_path");
    if (input_path.empty())
    {
        cerr << "Usage : " << argv[0] << " [<input_path>]" << endl;
        return -1;
    }


    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture capture(input_path); //try to open string, this will attempt to open it as a video file or image sequence
    if (!capture.isOpened()) //if this fails, try to open as a video camera, through the use of an integer param
        capture.open(atoi(input_path.c_str()));
    if (!capture.isOpened()) {
        cerr << "Failed to open the video device, video file or image sequence!\n" << endl;
        return 1;
    }

    // Setup output video
    cv::VideoWriter output_capture(output_path,
    													capture.get(CV_CAP_PROP_FOURCC),
															capture.get(CV_CAP_PROP_FPS),
															cv::Size(capture.get(CV_CAP_PROP_FRAME_WIDTH),
															capture.get(CV_CAP_PROP_FRAME_HEIGHT)));


    // import video
    vector<Mat> images;
    while(true){
    	Mat frame;
    	bool success = capture.read(frame);
    	if(!success)
    				break;
    	images.push_back(frame);
    }


    ifstream fin;
    fin.open(flows_path.c_str());
    // init optical_flow
    vector<Mat> flows;
    for (size_t t=1; t<images.size(); ++t) {

			if (fin.good())
				break;


		  Mat cur_grey;
		  Mat prev_grey;
		  cvtColor(images[t-1], prev_grey, COLOR_BGR2GRAY);
		  cvtColor(images[t], cur_grey, COLOR_BGR2GRAY);


	    double start = (double)getTickCount();
	    KLTFeatureMatch klt = KLTFeatureMatch();
	    klt.trackfeature(prev_grey, cur_grey);
	    Mat cur_grey_warp = klt.getWarpImage();
	    //Mat warp_image = klt.drawfeatureTrack();
	    //imshow("warp_image",warp_image);
	    waitKey(10);
	    double timeSec = (getTickCount() - start) / getTickFrequency();
	    cout << "calctrackfeature : " << timeSec << " sec" << endl;


	    double alpha = 0.025;
	    double ratio = 0.5;
	    int minWidth = 20;
	    int nOuterFPIterations = 10;
	    int nInnerFPIterations = 1;
	    int nSORIterations = 30;

	    DImage Im1(prev_grey.cols, prev_grey.rows, prev_grey.channels()),Im2(cur_grey_warp.cols, cur_grey_warp.rows, cur_grey_warp.channels());
	    ImageIO::MatToImage(prev_grey, Im1.data());
	    ImageIO::MatToImage(cur_grey_warp, Im2.data());

	    //Mat cur_warped;
	    //cv::warpPerspective(images[t], cur_warped, klt.getHmatrix(), images[t].size(), CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
	    //DImage Im1(images[t-1].cols, images[t-1].rows, images[t-1].channels()),Im2(images[t].cols, images[t].rows, images[t].channels());
	    //ImageIO::MatToImage(images[t-1], Im1.data());
	    //ImageIO::MatToImage(cur_warped, Im2.data());

	    DImage vx,vy,warpI2;
	    start = (double)getTickCount();
	    OpticalFlow::Coarse2FineFlow(vx, vy, warpI2, Im1, Im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations);
	    timeSec = (getTickCount() - start) / getTickFrequency();
	    cout << "Coarse2FineFlow : " << timeSec << " sec" << endl;


	    double* p_vx = vx.data();
	    double* p_vy = vy.data();

	    Mat point(prev_grey.size(), CV_32FC2);
	    for (int y=0; y<point.rows; ++y){
	    		for (int x=0; x<point.cols; ++x){
    				float flowx = p_vx[y*point.cols+x];
    				float flowy = p_vy[y*point.cols+x];
    				point.at<Point2f>(y, x) = Point2f((x + flowx), (y + flowy));
	    		}
	    }


	    	/*
      Mat flow;
      Ptr<DualTVL1OpticalFlow> tvl1 = cv::DualTVL1OpticalFlow::create();
      tvl1->setScaleStep(0.5);
      tvl1->setTau(0.25);
      tvl1->setWarpingsNumber(3);
      //tvl1->setTheta(0.3); // defalut 0.3
      //tvl1->setLambda(0.10); // default 0.15
      //cout<< tvl1->getWarpingsNumber() << endl;

      start = (double)getTickCount();
      tvl1->calc(prev_grey, cur_grey_warp, flow);
      timeSec = (getTickCount() - start) / getTickFrequency();
      cout << "calcOpticalFlowDual_TVL1 : " << timeSec << " sec" << endl;


      Mat point=flow.clone();
			for (int y = 0; y < flow.rows; ++y){
				for (int x = 0; x < flow.cols; ++x){
				Point2f f = point.at<Point2f>(y, x);
				point.at<Point2f>(y, x) = Point2f((x + f.x), (y + f.y));
				}
			}
			*/


			Mat out_flow;
			perspectiveTransform(point, out_flow, klt.getHmatrix());
			for (int y = 0; y < point.rows; ++y){
				for (int x = 0; x < point.cols; ++x){
				Point2f f = out_flow.at<Point2f>(y, x);
				out_flow.at<Point2f>(y, x) = Point2f((f.x-x), (f.y-y));
				}
			}

      flows.push_back(out_flow);

	    Mat temp;
	    //drawOpticalFlow(flow, temp);
	    //imshow("ori flow",temp);
	    //waitKey(10);

	    drawOpticalFlow(out_flow, temp);
	    imshow("initial flow",temp);
	    waitKey(10);


	    //vector<Mat> mask(1);
	    //vector<Mat> F={out_flow};
			//SteedyFlow sf = SteedyFlow();
			//sf.spatialAnalysis(F, mask);

			//imshow("mask",mask[0]);
			//waitKey(10);
    }


		if (fin.good()){
			flows = vecmatread(flows_path);
			cout << "read vector<Mat> from" << flows_path << endl;
		} else {
			vecmatwrite(flows_path, flows);
			cout << "write vector<Mat> to" << flows_path << endl;
		}
		fin.close();

		string comflows_path = flows_path;
		comflows_path.append("_compledted");


		//flows.resize(70);

		// start steedyflow

		vector<Mat> mask(flows.size());
		vector<Mat> completedFlows(flows.size());
		vector<Mat> F(flows.size());
		vector<Mat> C(flows.size());
		vector<Mat> ori_C(flows.size());
		vector<Mat> P(flows.size());
		vector<Mat> B(flows.size());
		vector<Mat> S(flows.size());

		for (size_t t=0; t<flows.size(); ++t){
			F[t] = flows[t].clone();
		}


		ori_C[0]=flows[0].clone();
		for(size_t t=1; t<(size_t)flows.size(); ++t){
			ori_C[t] = ori_C[t-1] + flows[t];
		}


		SteedyFlow sf = SteedyFlow();
		sf.spatialAnalysis(F, mask);
		//sf.setUseMaskBoundary(false);
		for (int i=0; i<5; ++i){

			fin.open(comflows_path.append(to_string(i)));
			if (fin.good()){
				completedFlows = vecmatread(comflows_path);
				cout << "read vector<Mat> from" << comflows_path << endl;
			} else {
				sf.completeMotion(F, completedFlows, mask, images);
				vecmatwrite(comflows_path, completedFlows);
				cout << "write vector<Mat> to" << comflows_path << endl;
			}
			fin.close();

			/*
			if (i==0){
				ori_C[0]=completedFlows[0].clone();
				for(size_t t=1; t<(size_t)flows.size(); ++t){
					ori_C[t] = ori_C[t-1] + completedFlows[t];
				}
			}
			*/
			sf.setUseMaskBoundary(true);

			for (size_t t=0; t<flows.size(); ++t){
				F[t] = completedFlows[t].clone();
			}

			//C[0]=flows[0].clone();
			C[0] = completedFlows[0].clone();
			for(size_t t=1; t<(size_t)flows.size(); ++t){
				//C[t] = C[t-1] + flows[t];
				C[t] = C[t-1] + completedFlows[t];
			}


			cout << "compute accflows"<<endl;


			if (i==0){
				P[0] = completedFlows[0].clone();
				for(size_t t=1; t<(size_t)flows.size(); ++t){
					P[t] = P[t-1] + completedFlows[t];
					}
			}

			S[0]=P[0].clone();
			for (size_t t=1; t<B.size(); ++t){
				S[t] = P[t]-P[t-1];
			}

			vector<Point2i> t_Windows;
			sf.computeTemporalWindow(S, t_Windows);
			sf.smoothFlow(C, P, t_Windows);
			//sf.smoothFlow(ori_C, P, t_Windows);

			for (size_t t=0; t<P.size(); ++t){
				//B[t] = P[t]-ori_C[t];
				B[t] = P[t]-C[t];
			}

			S[0]=P[0].clone();
			for (size_t t=1; t<B.size(); ++t){
				S[t] = P[t]-P[t-1];
			}

			sf.temporalAnalysis(P, mask);
			//sf.spatialAnalysis(S, mask);
			sf.setIter(sf.iter()+1);

			for (size_t t=0; t<P.size(); ++t){
				Mat dst;
				drawOpticalFlow(S[t], dst);
				imshow("S", dst);
				waitKey(10);


				Mat ref = (-1*B[t]);
				Mat map(P[t].size(), CV_32FC2);
				for (int y = 0; y < map.rows; ++y){
					for (int x = 0; x < map.cols; ++x){
						Point2f f = ref.at<Point2f>(y, x);
			      map.at<Point2f>(y, x) = Point2f((x + f.x), (y + f.y));
					}
				}

			  Mat out;
			  cv::remap(images[t+1], out, map, cv::Mat(), CV_INTER_LINEAR);
			  imshow("video", images[t]);
			  waitKey(10);
			  imshow("steedyflow result", out);
			  waitKey(10);
			  if (i==4)
				  output_capture.write(out);

			  imshow("mask_t", mask[t]);
			  waitKey(10);
			}
			cv::destroyAllWindows();
		}

    return 0;
}


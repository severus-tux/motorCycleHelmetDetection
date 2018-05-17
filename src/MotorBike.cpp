#include "MotorBike.h"

MotorBike::MotorBike(const Blob &blob) : Blob(blob)
{
	currentBoundingRect = blob.currentBoundingRect;
	currentDiagonalSize = blob.currentDiagonalSize;
	currentAspectRatio = blob.currentAspectRatio;
	directionLeft = blob.directionLeft; // true => left, false => Right
	crossTime = blob.crossTime;
}

int MotorBike::countRiders(cv::Mat &frame, cv::CascadeClassifier &head_cascade)
{
	int count=0;
	std::vector< cv::Rect > detections;
	std::vector< double > foundWeights;
	ROITop.x=currentBoundingRect.x;
	ROITop.y=currentBoundingRect.y;
	ROITop.width=currentBoundingRect.width;
	ROITop.height=(int) currentBoundingRect.height*0.25;
	cv::Mat ROI = frame(ROITop);
	//cv::cvtColor(ROI,ROI,CV_BGR2GRAY);
	head_cascade.detectMultiScale(ROI,detections,1.1,3,0, cv::Size(16,16), cv::Size(75,75)); 	
	
	//for ( size_t j = 0; j < detections.size(); j++ )
	//	if( foundWeights[j] >= 0.5 )
	//		count++;
	
	return detections.size();
}

int MotorBike::detectHelmet(cv::Mat &frame, cv::CascadeClassifier &helmet_cascade)
{
	int count=0;
	std::vector< cv::Rect > detections;
	std::vector< double > foundWeights;
	ROITop.x=currentBoundingRect.x;
	ROITop.y=currentBoundingRect.y;
	ROITop.width=currentBoundingRect.width;
	ROITop.height=(int) currentBoundingRect.height*0.25;
	cv::Mat ROI = frame(ROITop);
	//cv::cvtColor(ROI,ROI,CV_BGR2GRAY);
	helmet_cascade.detectMultiScale(ROI,detections,1.1,3,0, cv::Size(16,16), cv::Size(75,75)); 	
	
	//for ( size_t j = 0; j < detections.size(); j++ )
	//	if( foundWeights[j] >= 0.5 )
	//		count++;
	
	return detections.size();
}

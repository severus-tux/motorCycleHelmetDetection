#ifndef MOTORBIKE_H
#define MOTORBIKE_H

#include "Blob.h"

class MotorBike : public Blob
{
	public :
		cv::Rect ROITop;
		int riderCount;
		int HelmetCount;	
		std::vector< cv::Rect > detectionsHead;
		std::vector< cv::Rect > detectionsHelmet;	
		
		MotorBike(const Blob &blob);
		int countRiders(cv::Mat &frame, cv::CascadeClassifier &head_cascade);
		int detectHelmet(cv::Mat &frame, cv::CascadeClassifier &helmet_cascade);
};

#endif 

#ifndef MOTORBIKE_H
#define MOTORBIKE_H

#include "Blob.h"
#include "opencv2/core/core.hpp"


class MotorBike : public Blob
{
	public :
		cv::Rect ROI;
		int riderCount;
		int HelmetCount;		
		
		int countRiders();
		int detectHelmet();
		void extractROI();
};

#endif 

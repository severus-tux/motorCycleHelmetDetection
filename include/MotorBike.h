#ifndef MOTORBIKE_H
#define MOTORBIKE_H

#include "Blob.h"

class MotorBike : public Blob
{
	public :
		cv::Rect ROI;
		int riderCount;
		int HelmetCount;		
		
		MotorBike(const Blob &blob);
		int countRiders();
		int detectHelmet();
};

#endif 

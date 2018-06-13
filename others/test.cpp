#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml.hpp>
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_BLUE = cv::Scalar(255.0, 0.0, 0.0);
std::vector<std::vector<cv::Point> > contours;
cv::Mat frame, fgMask, fgMaskCopy, frameCopy;
cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
int main(int argc, char* argv[])
{
	char checkForEscKey = 0;
	if( argc != 2 )
	{
		std::cerr << "Sorry! incorrect usage!\n"
				  << "Usage : startDetection <path to surveillance feed>\n"
				  << "Example : startDetection video/traffic.mp4\n"
				  << "Example : startDetection http://192.168.1.102:8080/video\n";
		return 1;
	}
	cv::VideoCapture capVideo;
	cv::Ptr<cv::BackgroundSubtractor> fg;
	fg=cv::createBackgroundSubtractorMOG2(500,16,false);
	capVideo.open(argv[1]);
	if (!capVideo.isOpened())
	{                                    
		std::cerr << "error accessing video file/ remote camera\n" << std::endl << std::endl;
		return 1;                                                              
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2)
	{
		std::cerr << "error: video file must have at least two frames\n";
		return 1;
	}
	while (capVideo.isOpened())
	{
		capVideo.read(frame);
		frameCopy = frame.clone();
		cv::imshow("1", frame);
		cv::cvtColor(frameCopy, frameCopy, CV_BGR2GRAY);
		cv::imshow("2", frameCopy);
		cv::GaussianBlur(frameCopy, frameCopy, cv::Size(11, 11), 0);
		cv::imshow("3", frameCopy);
		fg->apply(frameCopy,fgMask,-1);
		cv::imshow("4", fgMask);
		cv::Mat fgMaskCopy = fgMask.clone();
		cv::morphologyEx(fgMaskCopy,fgMaskCopy,cv::MORPH_OPEN,structuringElement3x3,cv::Point(-1,-1),3,cv::BORDER_CONSTANT);
		cv::dilate(fgMaskCopy, fgMaskCopy, structuringElement5x5);
		cv::dilate(fgMaskCopy, fgMaskCopy, structuringElement5x5);
		cv::imshow("5", fgMaskCopy);
		cv::findContours(fgMaskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		std::vector<std::vector<cv::Point> > convexHulls(contours.size());
		for (unsigned int i = 0; i < contours.size(); i++)
		{
			cv::convexHull(contours[i], convexHulls[i]);
			cv::drawContours(frame, contours, i, SCALAR_BLUE, 2);
		}
		cv::imshow("6", frame);
		checkForEscKey = cv::waitKey(24);
		if(checkForEscKey == 27)
			break;
		if(checkForEscKey == (int) 'p')
		{
			while( checkForEscKey == (int) 'p' )
			{
				checkForEscKey = cv::waitKey(0);
				if(checkForEscKey == (int) 'p')
					break;
			}
		}
	}
return 0;
}

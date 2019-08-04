#include "tracker.h"
#include "boxextract.h"

//static const char* path = (char*)"../redcar/";
int main(int argc, char** argv)
{
	//cv::VideoCapture cap;
	BoxExtractor box;
	//assert(cap.open(path));
	cv::Mat frame;
	int initflag = 1;
	cv::Rect roi;
	int state;

	stcTracker* newtracker = (stcTracker*)malloc_tracker();
	char buf[128];
	int nums = 0;
	while(nums <= 1918)
	{
		sprintf(buf, "../redcar/%04d.jpg", ++nums);
		frame = cv::imread(buf);
		if(initflag == 1)
		{
			roi = box.extract(frame);
			stcTrackinit(newtracker, roi, frame);
			initflag = 0;
		}
		else
			stcTrackUpdate(newtracker, frame);
		state = getTrackState(newtracker);
		roi = getTrackRes(newtracker);
		if(!state)
			cv::rectangle(frame, roi, cv::Scalar(255,0,0));
		cv::imshow("tracking.jpg", frame);
		cv::waitKey(20);
	}
	free_track(newtracker);
	return 0;
}
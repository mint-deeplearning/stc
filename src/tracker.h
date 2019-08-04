#ifndef _TRACKER_H_HH
#define _TRACKER_H_HH

#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>

typedef struct _stctracker_param
{
	float padding;
	float rho;
	float scale;
	float lambda;
	int num;
	float alpha;
	float sigma;
	float eps;
	float failed_thr;
}param_t;

typedef enum{
	_wait = -1,
	_track,
	_lose,
}track_state;

typedef struct _stctracker{
	cv::Size _winsize;
	cv::Mat _dist;
	cv::Mat _confMap;
	cv::Mat _hann;
	cv::Mat _window;
	cv::Mat _contextprior;
	cv::Mat _Hstcf;
	cv::Rect _roi;
	param_t _param;
	int count;
	track_state _state;
}stcTracker;

void* malloc_tracker();
void free_track(stcTracker* tracker);

void stcTrackinit(stcTracker* newtracker, const cv::Rect roi, const cv::Mat & src);
void stcTrackUpdate(stcTracker* newtracker, const cv::Mat & src);

cv::Rect getTrackRes(stcTracker* newtracker);
int getTrackState(stcTracker* newtracker, bool silence = true);
#endif
#pragma once
#ifndef _EXTRACTOR_BOX_H_HH
#define _EXTRACTOR_BOX_H_HH

#include <opencv2/opencv.hpp>
using namespace cv;

class BoxExtractor {
public:
	Rect extract(Mat img);
	Rect extract(const std::string& windowName, Mat img, bool showCrossair = true);

	struct handlerT {
		bool isDrawing;
		Rect box;
		Mat image;

		// initializer list
		handlerT() : isDrawing(false) {};
	}params;

private:
	static void mouseHandler(int event, int x, int y, int flags, void *param);
	void opencv_mouse_callback(int event, int x, int y, int, void *param);
};
#endif


#include "tracker.h"
#include <string.h>

cv::Mat fft2(cv::Mat img, bool back = false)
{
	if(img.channels() == 1)
	{
		cv::Mat planes[] = {cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size())};
		cv::merge(planes, 2, img);
	}
	cv::dft(img, img, back ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0);
	return img;
}

void set_default_param(param_t* param)
{
	assert(param != NULL);
	param->padding = 1.0;
	param->rho = 0.075;
	param->scale = 1.0;
	param->lambda = 0.25;
	param->num = 7;
	param->alpha = 2.25;

	param->eps = 0.0000001;
	param->failed_thr = 0.35;
}

cv::Mat cal_confmap(int sizex, int sizey, float alpha, cv::Mat & dist)
{
	int ry = sizey / 2;
	int rx = sizex / 2;
	int r,c;
	cv::Mat conf = cv::Mat::zeros(sizey, sizex, CV_32F);
	float sum = 0.0;
	for(r = 0; r < sizey; r++)
		for(c = 0; c < sizex; c++)
		{
			int difx = c - rx;
			int dify = r - ry;
			dist.at<float>(r,c) = difx * difx + dify * dify;
			float val = std::sqrt(dist.at<float>(r,c));
			conf.at<float>(r,c) = exp(-0.5 / alpha * val);
		}
	return fft2(conf);
}

cv::Mat createHanningWindow(const cv::Size & size)
{
	cv::Mat hann1 = cv::Mat::zeros(1, size.width, CV_32FC1);
	cv::Mat hann2 = cv::Mat::zeros(size.height, 1, CV_32FC1);
	for(int i = 0; i < hann1.cols; i++)
		hann1.at<float>(0,i) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann1.cols - 1)));
	for(int i = 0; i < hann2.rows; i++)
		hann2.at<float>(i,0) = 0.5 * (1 - std::cos(2 * CV_PI * i / (hann2.rows - 1)));
	return hann2 * hann1;
}

cv::Mat cal_window(const cv::Mat & han, float sigma, const cv::Mat & dist)
{
	cv::Mat temp = cv::Mat::zeros(dist.size(), CV_32FC1);
	int r,c;
	for(r = 0; r < temp.rows; r++)
		for(c = 0; c < temp.cols; c++)
		{
			float val = dist.at<float>(r,c);
			temp.at<float>(r,c) = std::exp(-0.5 / (sigma * sigma) * val);
		}
	cv::Mat window = han.mul(temp);
	return window;
}

cv::Mat get_context(const cv::Mat & src, cv::Rect roi, cv::Size _winsize, const cv::Mat & window)
{
	float cx = roi.x + 0.5 * roi.width;
	float cy = roi.y + 0.5 * roi.height;
	cv::Rect extracted_roi;
	extracted_roi.width = _winsize.width;
	extracted_roi.height = _winsize.height;
	extracted_roi.x = cx - 0.5 * extracted_roi.width;
	extracted_roi.y = cy - 0.5 * extracted_roi.height;
	cv::Rect res = extracted_roi & cv::Rect(0,0,src.cols,src.rows);
	assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);

	cv::Rect border;
	border.x = res.x - extracted_roi.x;
	border.y = res.y - extracted_roi.y;
	border.width = extracted_roi.x + extracted_roi.width - (res.x + res.width);
	border.height = extracted_roi.y + extracted_roi.height - (res.y + res.height);

	cv::Mat _res;
	src(res).convertTo(_res,CV_32F);
	if(border != cv::Rect(0,0,0,0))
	{
		cv::copyMakeBorder(_res,_res,border.y,border.height,border.x,border.width,cv::BORDER_REPLICATE);
	}
	float mean_val = (float)cv::mean(_res)[0];
	_res -= mean_val;
	_res = window.mul(_res);
	return _res;
}

cv::Mat set_mat_to_8uc1(cv::Mat src)
{
	cv::Mat gray;
	if(src.type() == CV_8UC3)
	{
		cv::cvtColor(src, gray, CV_BGR2GRAY);
		return gray;
	}
	return src;
}

cv::Mat complexDivide(const cv::Mat & x1, const cv::Mat & x2)
{
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(x1, pa);
	cv::split(x2, pb);
	cv::Mat a = pa[0];
	cv::Mat b = pa[1];
	cv::Mat c = pb[0];
	cv::Mat d = pb[1];
	cv::Mat coeff = 1.0 / (c.mul(c) + d.mul(d));
	std::vector<cv::Mat> planes;
	planes.push_back((a.mul(c) + b.mul(d)).mul(coeff));
	planes.push_back((b.mul(c) - a.mul(d)).mul(coeff));
	
	cv::Mat divide;
	cv::merge(planes, divide);
	return divide;
}

cv::Mat complexmultiplication(cv::Mat x1, cv::Mat x2)
{
	std::vector<cv::Mat> pa;
	std::vector<cv::Mat> pb;
	cv::split(x1, pa);
	cv::split(x2, pb);

	std::vector<cv::Mat> pres;
	pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
	pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

	cv::Mat res;
	cv::merge(pres, res);
	return res;
}

cv::Mat real(cv::Mat src)
{
	std::vector<cv::Mat> planes;
	cv::split(src, planes);
	return planes[0];
}

void stcTrackprepare(stcTracker* newtracker, const cv::Rect roi)
{
	assert(newtracker != NULL);

	set_default_param(&newtracker->_param);
	cv::Rect _roi = roi;
	newtracker->_roi = roi;
	newtracker->count = 0;
	newtracker->_state = _wait;
	newtracker->_param.sigma = 0.5 * (_roi.width + _roi.height);

	newtracker->_winsize.width = _roi.width * (1 + newtracker->_param.padding);
	newtracker->_winsize.height = _roi.height * (1 + newtracker->_param.padding);

	newtracker->_dist = cv::Mat::zeros(newtracker->_winsize, CV_32FC1);
	newtracker->_confMap = cal_confmap(newtracker->_winsize.width, newtracker->_winsize.height,\
							newtracker->_param.alpha, newtracker->_dist);

	newtracker->_hann = createHanningWindow(newtracker->_winsize);

	newtracker->_window = cal_window(newtracker->_hann, newtracker->_param.sigma, newtracker->_dist);

}

void stcTracktrain(stcTracker* newtracker, const cv::Mat & mat_8uc1, float rho)
{
	if(newtracker->count)
		newtracker->_contextprior = get_context(mat_8uc1, newtracker->_roi, newtracker->_winsize, newtracker->_window);
	cv::Mat hscf;
	hscf = complexDivide(newtracker->_confMap, fft2(newtracker->_contextprior) + newtracker->_param.eps);
	newtracker->_Hstcf = (1 - rho) * newtracker->_Hstcf + rho * hscf;
}

void stcTrackinit(stcTracker* newtracker, const cv::Rect roi, const cv::Mat & src)
{
	stcTrackprepare(newtracker, roi);

	cv::Mat track_mat = set_mat_to_8uc1(src);

	newtracker->_contextprior = get_context(track_mat, newtracker->_roi, newtracker->_winsize, newtracker->_window);

	newtracker->_Hstcf = cv::Mat::zeros(newtracker->_winsize, CV_32FC2);

	stcTracktrain(newtracker, track_mat, 1.0);
}

void stcTrackUpdate(stcTracker* newtracker, const cv::Mat & src)
{
	newtracker->_param.sigma *= newtracker->_param.scale;
	newtracker->_window = cal_window(newtracker->_hann, newtracker->_param.sigma, newtracker->_dist);
	newtracker->count = 1;
	cv::Mat track_mat = set_mat_to_8uc1(src);
	newtracker->_contextprior = get_context(track_mat, newtracker->_roi, newtracker->_winsize, newtracker->_window);

	cv::Mat response = real(fft2(complexmultiplication(newtracker->_Hstcf, fft2(newtracker->_contextprior)), true));
	double peak_val;
	cv::Point pi;
	cv::minMaxLoc(response, NULL, &peak_val, NULL, &pi);
	printf("peak val: %.3f\n", peak_val);

	if(peak_val > newtracker->_param.failed_thr)
	{
		newtracker->_state = _track;
		newtracker->_roi.x = newtracker->_roi.x - newtracker->_winsize.width / 2 + pi.x;
		newtracker->_roi.y = newtracker->_roi.y - newtracker->_winsize.height / 2 + pi.y;
	}
	else{
		newtracker->_state = _lose;
		return;
	}
	stcTracktrain(newtracker, track_mat, newtracker->_param.rho);
}

cv::Rect getTrackRes(stcTracker* newtracker)
{
	return newtracker->_roi;
}

int getTrackState(stcTracker* newtracker, bool silence)
{
	int state = (int)newtracker->_state;
	if(!silence)
	{
		switch(state)
		{
			case -1:
				printf("waiting state!\n");
				break;
			case 0:
				printf("tracking state!\n");
				break;
			case 1:
				printf("lose state!\n");
				break;
			default:
				printf("unknow state!\n");
				break;
		}
	}
	return state;
}

void* malloc_tracker()
{
	stcTracker* newtracker = new stcTracker;
	static int val = -1;
	if(newtracker == NULL){
		printf("alloc memory failed!\n");
		return (void*)&val;
	}
	memset(newtracker,0,sizeof(stcTracker));
	return (void*)newtracker;
}

void free_track(stcTracker* tracker)
{
	if(tracker != NULL)
		delete tracker;
}
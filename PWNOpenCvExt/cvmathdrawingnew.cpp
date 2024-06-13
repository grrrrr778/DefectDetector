// (c) Pawlin Technologies Ltd 2019
// File: cvmathdrawingnew.h, .cpp
// Purpose: file for opencv extensions code for mathematics and visualizations
// Author: P.V. Skribtsov
// ALL RIGHTS RESERVED
#include "stdafx.h"
#include "cvmathdrawingnew.h"
#include <PWNImageObj/ConcentrationEllipseEstimator.h>
#include <PWNProfiler/duration.h>
#include <PWNGeneral/TXTreader.h>
namespace pawlin {

	void pawlin::PolyModel1D::testPoly()
	{
		Dataset data;
		const float vel = 0.5f;
		const float acc = 0.02f;
		const float maxt = 1.0f;
		const int samples = 10;
		for (int i = 0; i < samples; i++) {
			float t = maxt*i / (samples - 1);
			FloatVector row;
			row.push_back(t);
			row.push_back(100.0f + vel*t + acc*t*t / 2.0f);
			data.push_back(row);
			printf("time %f, pos %f\n", t, row.data.back());
		}
		Profiler prof;
		prof.startSequence();
		pawlin::PolyModel1D model(data, 2);
		prof.markPoint("compute poly");
		prof.print();
		model.getCoefs().print();
		for (int i = 0; i < samples; i++) {
			float t = maxt*i / (samples - 1);
			printf("t = %f \t pos %f\t", t, model.compute(t));
			printf("vel %f\t", model.computeD(t));
			printf("acc %f\n", model.computeD2(t));
		}
	}
	//void stitch(const vector <cv::Mat> &imgs, const string &result_name = "result.jpg") {
	//	bool try_use_gpu = false;
	//	bool divide_images = false;
	//	cv::Stitcher::Mode mode = cv::Stitcher::SCANS;
	//	cv::Mat pano;
	//	cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode, try_use_gpu);
	//	cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
	//	cv::imwrite(result_name, pano);
	//	printf("status %d\n", status);
	//}

//	void drawOvalBase(const OvalBase &oval, cv::Scalar color, int width, float scaleX, float scaleY, cv::Mat &img, bool drawAxis) {
//		cv::RotatedRect box;
//		
//		box.center = cv::Point2f(oval.x, oval.y);
//		box.angle = oval.angle + 90.0f; // need to add 90.0f if oval was normalized for correct display...
//#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION == 1
//										// do nothing
//#else
//		box.angle *= -1.0f;
//#endif
//		box.size = cv::Size2f(oval.sx*scaleX, oval.sy*scaleY);
//		cv::ellipse(img, box, color, width, cv::LINE_AA);
//		cv::Point c((int)oval.x, (int)oval.y);
//		cv::Point e1(point(OvalBase::Basis(oval).e1));
//		cv::Point e2(point(OvalBase::Basis(oval).e2));
//		if (drawAxis) {
//			cv::arrowedLine(img, c, c + e1, color);
//			cv::arrowedLine(img, c, c + e2, color);
//		}
//	}

	void CursorHandler::run(const std::string &winname, const cv::Mat & background)
	{
		cursor = background.cols / 2;
		cv::namedWindow(winname);
		cv::setMouseCallback(winname, onMouse, this);
		for (;;) {
			cv::Mat canvas = background.clone();
			int i = cursor;

			cv::line(canvas, cv::Point(i, 0), cv::Point(i, background.rows), CV_RGB(255, 255, 255), 1); // draw cursor
			cv::imshow(winname, canvas);

			int k = cv::waitKeyEx(10);

			if (k == OPENCV_RIGHT_ARROW_CODE) cursor = (cursor + 1) % background.cols;
			if (k == OPENCV_LEFT_ARROW_CODE) cursor = (cursor - 1) % background.cols;
			if (cursor < 0) cursor = 0;
			char key = 0; if((k&0xFF00)==0) key= k & 255;
			if (key <= 0) continue;
			if (key == 27) break;
			onColumn(cursor, Event::KEY, k);
		}
	}

	void CursorHandler::onMouse(int event, int x, int y, int flags, void * param)
	{
		if (event == cv::MouseEventTypes::EVENT_LBUTTONDOWN) {
			CursorHandler *_this = (CursorHandler *)param;
			_this->cursor = x;
			_this->onColumn(x, CursorHandler::Event::CLICK, flags);
		}
	}



	void CvInteractWindowBase::track_callback(int pos, void *params) {
		CvInteractWindowBase *_this = (CvInteractWindowBase *)params;
		_this->updateSliders();
	}

	void CvInteractWindowBase::mouse_callback(int event, int x, int y, int flags, void *params) {
		CvInteractWindowBase *_this = (CvInteractWindowBase *)params;
		_this->updateClick(x, y, event, flags);
	}

	void imshowWithResize(string windowName, const cv::Mat &img, int sizeMul, int cvinterpolation) {
		cv::Mat resized;
		cv::resize(img, resized, cv::Size(img.cols*sizeMul, img.rows*sizeMul), 0.0, 0.0, cvinterpolation);
		//cv::normalize(resized,resized,0.0,1.0,NORM_MINMAX);
		cv::imshow(windowName, resized);
	}

	void SelectionSolidROIWindow::show(cv::Mat &canvas) const {
		SelectionROIWindow::show(canvas);
		cv::putText(canvas, message, cv::Point(segment.first.x + 10, segment.second.y + 20), cv::FONT_HERSHEY_PLAIN, 1, framecolor, 1);
	}

	cv::Rect SelectionSolidROIWindow::getROI() const {
		return cv::Rect(segment.first, size);
	}

	void MultiRectSelectionWindow::test()
	{
		cv::Mat temp(480, 640, CV_8UC3);
		temp = CV_RGB(100, 150, 250);
		vector <string> labels = { "GOOD","MID","BAD" };
		vector <cv::Scalar> colors = { CV_RGB(0,255,0), CV_RGB(200,150,0), CV_RGB(255,0,0) };
		MultiRectSelectionWindow obj("MultiRectSelectionWindow", temp, labels, colors);
		obj.run();
	}

	cv::Mat buildMask4Color(const cv::Mat &mask8UC3, cv::Scalar color) {
		cv::Mat mask = mask8UC3.clone();
		cv::Mat mask_float;
		mask.convertTo(mask_float, CV_32F);
		//std::map<Point3Df, int> m;
		//buildMatHist(mask_float, m, false);
		//Point3Df test((float)color[0], (float)color[1], (float)color[2]);
		//printf("image has %d pixels of the color (%f,%f,%f)\n",
		//	m[test],test.x,test.y,test.z
		//	);
		//debugImg("mask8UC3", mask, 1,-1,false);
		//debugImg("mask_float", mask_float*(1.0/255.0), 1,0, false);
		mask_float = (mask_float - color);
		vector<cv::Mat> channels;
		cv::split(mask_float, channels);
		return channels[0] == 0 & channels[1] == 0 & channels[2] == 0;
	}

	CosmicPanel::CosmicPanel(
		const std::string &winname,
		const std::string &panelfilename,
		float scale,
		const cv::Scalar &flcolor, //flash when pressed
		const cv::Scalar &hlcolor //highlight
	) :
		highlight_color(hlcolor),
		flash_color(flcolor),
		CvInteractWindowBase(winname, cv::Size(640, 480), CV_8UC3)
	{
		panel = cv::imread(panelfilename);
		btnmask = cv::imread(panelfilename + ".mask.bmp");
		if (panel.size().area() == 0)
		{
			std::cerr << "ERROR: panelfile is not found: " << panelfilename << std::endl;
			abort();
		}
		smartScale(panel, panel, scale);
		smartScale(btnmask, btnmask, scale);
		StrReader str((panelfilename + ".csv").c_str(), ";\t");
		FOR_ALL(str, i) {
			if (str[i].size() != 5) throw std::runtime_error("CosmicPanel colormap config file has wrong format, must be (id;R;G;B;name)");
			Region r;
			r.id = atoi(str[i][0].c_str());
			r.color[2] = atoi(str[i][1].c_str()); //mind RGB->BGR representation in memory
			r.color[1] = atoi(str[i][2].c_str());
			r.color[0] = atoi(str[i][3].c_str());
			r.key = str[i][4];
			regions.push_back(r);
		}
		this->canv_size = panel.size();
		FOR_ALL(regions, c) {
			//printf("Build mask[%zu] with color %f,%f,%f\n",
			//	c,
			//	colorkeys[c][0],
			//	colorkeys[c][1],
			//	colorkeys[c][2]
			//);
			regions[c].colormap = buildMask4Color(btnmask, regions[c].color);
			regions[c].boundingBox = pawlin::boundingBox(regions[c].colormap);
			//debugImg("mask" + int2str(c), mask, 1);
		}
		ready = true;
	}
	void CosmicPanel::updateClick(int x, int y, int event, int mouseflags) {
		if (!ready) return; // do not allow to get events before construction is over
		cv::Point p(x, y);
		if (regions.size() && pawlin::imageRect(regions.front().colormap).contains(p)) {
			cursor = -1;
			FOR_ALL_IF(regions, c, (regions[c].clickable() && regions[c].colormap.at<uint8_t>(p))) cursor = (int)c;
			//printf("Cursor %d\n", cursor);
			if ((event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_LBUTTONDBLCLK) && cursor >= 0) {
				if (!flash) button_hit(regions[cursor].id, regions[cursor].key);
				flash = true;
			}
			if(event == cv::EVENT_LBUTTONUP) flash = false;
		}
	};
	void CosmicPanel::draw(cv::Mat &where) const {
		cv::Mat temp = panel + CV_RGB(5, 5, 5);
		cv::Mat highlight = cv::Mat(panel.size(), CV_32FC3);
		if (cursor >= 0) {
			highlight = CV_RGB(1, 1, 1);
			cv::Mat img = regions[cursor].colormap.clone();
			highlight.setTo(flash ? flash_color : highlight_color, img);
			cv::blur(highlight, highlight, cv::Size(15, 15));
			blend(highlight, temp);
			where = highlight;
		}
		else where = temp;
	}
	int CosmicPanel::show(int delay) const {
		cv::Mat temp;
		draw(temp);
		int key = debugImg(winname, temp /*cursor>=0 ? highlight : temp*/, 1, delay, false);
		return key;
	}

	 cv::Rect boundingBox(const cv::Mat &mask8UC1) {
		MinMaxAvg statx, staty;
		FOR_MAT(mask8UC1, p) {
			if (mask8UC1.at<uint8_t>(p)) {
				statx.take((float)p.x);
				staty.take((float)p.y);
			}
		}
		return cv::Rect(
			float2int(statx.minv),
			float2int(staty.minv),
			float2int(statx.maxv - statx.minv),
			float2int(staty.maxv - staty.minv)
		);
	}
	

} // end of namespace

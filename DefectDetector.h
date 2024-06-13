#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <set>
#include <vector>
#include <iterator>    
#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <PWNGeneral/pwnutil.h>
#include <PWNGeometry/Point3Df.h>
#include <PWNGeometry/Plane3Df.h>
#include <PWNStereo/DepthProcessor.h>
#include <PWNOpenCvExt/cvdrawingcompat.h>
#include <PWNGeneral/ArgParser.h>

#include "ImagePreparator.h"

#define OFFICE

class DefectDetector {
    ArgParser &parser;
    struct DataForCalculations {
        cv::Mat image;
        DepthProcessor proc;
        std::vector<Point3Df> points;
        std::string window;
        DataForCalculations(cv::Mat &im, std::string word) {
            image = im;
            proc.setIntrinsic(420.f, 420.f, 320.f, 240.f);
            window = word;
        }
    };
    AppParams app_params_;
public:

    DefectDetector(ArgParser &parser);
    void run();
private:
    void loadAndProcessAllFiles();
    void proccesFile(std::string filename);
    void prepareDepth(std::__cxx11::string &filename, cv::Mat &gray_blured);

    void filterContours(std::vector<std::vector<cv::Point>>* contours);
    void showDistance(int action, int x, int y, int flags, void * userdata);
    float medianMat(cv::Mat Input);
    cv::Mat getRidOfBlack(cv::Mat original);
    cv::Mat getBlured(cv::Mat original);
    std::string numberToText(float number);
    void saveSomeData(cv::Mat & image, Plane3Df & plane, cv::Mat & dist, cv::Mat & blured_dist);
    bool printDistance(int & it, cv::Rect bounding_rect, DepthProcessor & proc, float plane_min_point, float contour_area, cv::Mat & contour_mask, cv::Mat & image_for_dist, cv::Mat & to_show);
    void getImageWithMeasures(std::vector<std::vector<cv::Point>>& contours, cv::Mat & gray_copy, float min_point, bool & hole_exist, cv::Mat & gray_image_for_showing);
};

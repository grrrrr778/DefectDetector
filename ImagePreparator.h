#pragma once
#include <opencv2/core.hpp>
#include <PWNStereo/DepthProcessor.h>
#include <PWNGeometry/Point3Df.h>
#include <PWNGeometry/Plane3Df.h>

#include "AppParams.h"

class ImagePreparator {
    cv::Mat original_;
    cv::Mat for_showing_;
    cv::Mat postprocessed_;
    cv::Mat dif_depth_;
    Plane3Df plane_;
    std::vector<Point3Df> original_point_cloud_;
    std::vector<Point3Df> cloud_point_;
    std::vector<Point3DfColor> color_cloud_point_;
    std::vector<Point3Df> plane_point_cloud_;
    float min_point_;

    AppParams &app_params_;
public:
    ImagePreparator(cv::Mat depth_image, AppParams &app_params);
    ImagePreparator(std::string filename_depth_image, AppParams &app_params);

    bool getCloudPoint(std::vector<Point3Df> &point_cloud);
    bool getColorCloudPoint(std::vector<Point3DfColor> &color_point_cloud);
    bool getPostprocessedDepth(cv::Mat &processed_depth);
    bool getOriginalDepth(cv::Mat &original_depth);
    void getPlane(Plane3Df &processed_plane);
    float getDifferenceDepth(cv::Mat &dif_depth);
    void saveAllData();
private:
    void retrieveAllData();
    void retrieveCloudPoint(cv::Mat &input, std::vector<Point3Df> &output_cloud);
    void retrieveCloudPointFromPlane();
    void retrieveColorCloudPoint(cv::Mat &input);
    void retrieveDepthFromFile(std::string &filename);
    void retrievePostprocessedDepth();
    void retrievePlane();
    float retrieveDifferenceDepth(Plane3Df &plane, cv::Mat &real_depth);
    void getRidOfBlack(cv::Mat &input);

};

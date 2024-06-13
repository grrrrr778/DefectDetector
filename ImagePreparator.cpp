#include "ImagePreparator.h"

ImagePreparator::ImagePreparator(cv::Mat depth_image, AppParams &app_params) :app_params_(app_params) {
    original_ = depth_image.clone();
    min_point_ = MAXFLOAT;
    retrieveAllData();
}

ImagePreparator::ImagePreparator(std::string filename_depth_image, AppParams &app_params) : app_params_(app_params) {
    retrieveDepthFromFile(filename_depth_image);
    min_point_ = MAXFLOAT;
    retrieveAllData();
}

bool ImagePreparator::getCloudPoint(std::vector<Point3Df> &point_cloud) {
    if (cloud_point_.empty()) {
        printf("Empty point cloud\n");
        return false;
    }
    point_cloud = cloud_point_;
    return true;
}

bool ImagePreparator::getColorCloudPoint(std::vector<Point3DfColor> &color_point_cloud) {
    if (color_cloud_point_.empty()) {
        printf("Empty color point cloud\n");
        return false;
    }
    color_point_cloud = color_cloud_point_;
    return true;
}

bool ImagePreparator::getPostprocessedDepth(cv::Mat &processed_depth) {
    if (postprocessed_.empty()) {
        printf("Empty processed depth");
        return false;
    }
    processed_depth = postprocessed_.clone();
    return true;
}

bool ImagePreparator::getOriginalDepth(cv::Mat & original_depth) {
    if (original_.empty()) {
        printf("Empty original depth");
        return false;
    }
    original_depth = original_.clone();
    return true;
}

void ImagePreparator::getPlane(Plane3Df & processed_plane) {
    processed_plane = plane_;
}

float ImagePreparator::getDifferenceDepth(cv::Mat & dif_depth) {
    if (dif_depth_.empty()) {
        printf("Empty dif depth\n");
        return MAXFLOAT;
    }
    dif_depth = dif_depth_.clone();
    return min_point_;
}

void ImagePreparator::saveAllData() {

    FileSaver<Point3DfColor>::save("color_depth.3dcp", color_cloud_point_);
    FileSaver<Point3Df>::save("plane.3dp", plane_point_cloud_);
    FileSaver<Point3Df>::save("cloud.3dp", cloud_point_);
    FileSaver<Point3Df>::save("original_cloud.3dp", original_point_cloud_);
}

void ImagePreparator::retrieveAllData() {
    retrievePostprocessedDepth();
    retrieveCloudPoint(original_, original_point_cloud_);
    retrieveCloudPoint(postprocessed_, cloud_point_);
    retrieveColorCloudPoint(postprocessed_);
    retrievePlane();
    retrieveCloudPointFromPlane();
}

void ImagePreparator::retrieveCloudPoint(cv::Mat &input, std::vector<Point3Df> &output_cloud) {
    for (int i = 0; i < app_params_.width - app_params_.pts_per_frame; i += app_params_.pts_per_frame) {
        for (int j = 0; j < app_params_.height - app_params_.pts_per_frame; j += app_params_.pts_per_frame) {
            //++counter;
            Point3Df p(i, j, input.at<float>(j, i));
            //if (std::abs(p.x) < 0.1 || std::abs(p.y) < 0.1 || std::abs(p.z) < 0.1) continue;
            output_cloud.push_back(p);
        }
    }
    //std::vector<Point3Df> vp_for_save;
    //for (auto item : vp) {
    //    vp_for_save.push_back(Point3Df(item.x / 100.f, item.y / 100.f, -2 * item.z));
    //}
    //imshow("check", gray_copy);
    //FileSaver<Point3Df>::save("cloud_for_plane.3dp", vp_for_save);
}

void ImagePreparator::retrieveCloudPointFromPlane() {
    for (int i = 0; i < app_params_.width; i += app_params_.pts_per_frame)
    {
        for (int j = 0; j < app_params_.height; j += app_params_.pts_per_frame)
        {
            plane_point_cloud_.push_back(Point3Df((float)i, (float)j, plane_.computeZ(i, j)));

        }
    }
}

void ImagePreparator::retrieveColorCloudPoint(cv::Mat &input) {
    cv::Mat img_8U;
    cv::normalize(input, img_8U, 0, 1, cv::NORM_MINMAX);
    img_8U.convertTo(img_8U, CV_8UC1, 255);
    for (int i = 0; i < app_params_.width; i += app_params_.pts_per_frame)
    {
        for (int j = 0; j < app_params_.height; j += app_params_.pts_per_frame)
        {
            if (img_8U.at<uchar>(j, i) == 0) continue;
            int dep = img_8U.at<uchar>(j, i);
            color_cloud_point_.push_back(Point3DfColor(Point3Df(
                i / 100.f, j / 100.f, input.at<float>(j, i)),
                Point3Df(
                    1 - (float)dep / 256,
                    0.5f,
                    (float)dep / 256)
            ));
        }
    }
}

void ImagePreparator::retrieveDepthFromFile(std::string &filename) {
    original_ = imread(filename, cv::IMREAD_ANYDEPTH);
    original_.convertTo(original_, CV_32F);
}

void ImagePreparator::retrievePostprocessedDepth() {
    postprocessed_ = original_.clone();
    postprocessed_ = postprocessed_ * app_params_.depth_units;
    getRidOfBlack(postprocessed_);
}

void ImagePreparator::retrievePlane() {
    if (cloud_point_.empty()) return;
    plane_.setThickness(0.005); //0.5 cm
    float percent = plane_.approximateRANSAC(cloud_point_, 0.05f, 10000);
    printf("plane percent = %f", percent);
}

float ImagePreparator::retrieveDifferenceDepth(Plane3Df & plane, cv::Mat &real_depth) {
    float min_point_in_plane = 100;
    for (int i = 0; i < app_params_.width; i++)
    {
        for (int j = 0; j < app_params_.height; j++)
        {
            if (std::abs(real_depth.at<float>(j, i)) < 0.1) continue; //changed from 1e-6 to 0.1
            if (plane.computeZ(i, j) < min_point_in_plane) min_point_in_plane = plane.computeZ(i, j);
            real_depth.at<float>(j, i) = plane.computeZ(i, j) - real_depth.at<float>(j, i);
        }
    }
    return min_point_in_plane;
}

void ImagePreparator::getRidOfBlack(cv::Mat &input) {
    cv::Mat image;
    input.copyTo(image);
    cv::Mat blured;
    cv::Mat mask = image < 1;
    int count = 0;
    while (count < 5) {
        medianBlur(image, blured, 5);
        auto res = sum(mask);
        blured.copyTo(image, mask);
        mask = image < 1;
        ++count;
    }
    input = image.clone();
}
